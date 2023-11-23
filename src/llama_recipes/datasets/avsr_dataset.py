import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import random
import torch

import cv2 as cv
from torch.nn.utils.rnn import pad_sequence

class AVSRDataset(Dataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()

        self.modal = dataset_config.modal
        self.dataset = split                        #train|val|test
        self.data_path = dataset_config.data_path
        self.h5file = dataset_config.h5file   
        self.noiseFile = dataset_config.noiseFile
        self.noiseSNR =  dataset_config.noiseSNR
        self.noiseProb = dataset_config.noiseProb
        self.stepSize = dataset_config.stepSize  #16384
        self.charToIx = dataset_config.charToIx

        if self.dataset == "train": 
            pretrain_dir = self.data_path + "LRS3/pretrain.txt"
            train_dir = self.data_path + "LRS3/train.txt"

            with open(pretrain_dir, "r") as f:
                lines = f.readlines()
                pretrain_datalist = [self.data_path + line.strip()[3:] for line in lines] #长度：118516

            with open(train_dir, "r") as f:
                lines = f.readlines()
                train_datalist = [self.data_path + line.strip()[3:] for line in lines] #长度:31662

            self.datalist = pretrain_datalist+ train_datalist
            lrs3Aug=True

        elif self.dataset == "val":
            val_dir = self.data_path + "LRS3/val.txt"
            with open(val_dir, "r") as f:
                lines = f.readlines()
                val_datalist = [self.data_path + line.strip()[3:] for line in lines]
            self.datalist = val_datalist
            lrs3Aug=False

        else:
            test_dir = self.data_path + "LRS3/test.txt"
            with open(test_dir, "r") as f:
                lines = f.readlines()
                test_datalist = [self.data_path + line.strip()[3:] for line in lines]
            self.datalist = test_datalist
            lrs3Aug=False

        with h5py.File(self.noiseFile, "r") as f:  #{'noiseFile': '/home/xcpan/LRS2/mvlrs_v1/Noise.h5', 'noiseProb': 0.25, 'noiseSNR': 5}
            self.noise = f["noise"][0]  #ndarray:57600000

        if lrs3Aug:
            self.transform = transforms.Compose([
                ToTensor(),
                RandomCrop(112),
                RandomHorizontalFlip(0.5),
                Normalize(mean=[0.4161], std=[0.1688])
            ])
        else:
            self.transform = transforms.Compose([
                ToTensor(),
                CenterCrop(112),
                Normalize(mean=[0.4161], std=[0.1688])
            ])

    def open_h5(self):
        self.h5 = h5py.File(self.h5file, "r")

    def __getitem__(self, index):  #avsr 是shuffle的dataloader echat好像默认false 没shu  index从0开始
        """
            LRS3 : pretrain 118516  train 31662  val 320   test 1321
            LRS2 : pretrain 96318   train 45839  val 1082  test 1243    142157 = 96318 + 45839 = pretrain + train  143239 = 96318+45839+1082=pretrain+train+val

            index goes from 0 to stepSize-1
            dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            fetch the sample at position 'index' in this randomly selected partition
        """ 

        if not hasattr(self, 'h5'):
            self.open_h5()

        if self.dataset == "train":   #index=610
            base = self.stepSize * np.arange(int(len(self.datalist) / self.stepSize) + 1)   # datalist, 118516 应该全是pretrain的 从pretrain.txt 搞出来的 # stepsize 16384
            ixs = base + index                        # [  0  16384  32768  49152  65536  81920  98304 114688 131072 147456]
            ixs = ixs[ixs < len(self.datalist)]       # [  610  16994  33378  49762  66146  82530  98914 115298]
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)  #以某种方式随机采样  #33378

        if index==99639 or index== 71740 or index==19753 or index==14116 or index==49729 or index==26726:  #dirty data
            index+=1

        # passing the sample files and the target file paths to the prepare function to obtain the input tensors
        targetFile = self.datalist[index] + ".txt"  
        if self.dataset == "val":
            index += 150178             # 原本 142157 
        elif self.dataset == "test":
            index += 150498             # 原本 143239

        if np.random.choice([True, False], p=[self.noiseProb, 1 - self.noiseProb]):
            noise = self.noise
        else:
            noise = None

        if index < 118516:     #原本是96318   查过了 这个数确实是lrs2的那个行数 也就是文件数  原本应该是pretrain处理的 有一部分搞到main处理了 所以没有crop 导致超过500
            inp, trgtin, trgtout, trgtLen = self.prepare_pretrain_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)
            if inp==0 and trgtin ==0 and  trgtout ==0 and trgtLen==0:
                index+=1
                targetFile = self.datalist[index] + ".txt"
                inp, trgtin, trgtout, trgtLen = self.prepare_pretrain_input(index, self.modal, self.h5, targetFile,self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)  #就只是往后挪了一格 很弱

        else:
            inp, trgtin, trgtout, trgtLen = self.prepare_main_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR)

        return inp, trgtin, trgtout, trgtLen   #VO (none,(72,1,112,112) )

    def __len__(self):
        """
            each iteration covers only a random subset of all the training samples whose size is given by the step size   step size的作用在这里 感觉也没什么大用
            this is done only for the pretrain set, while the whole val/test set is considered
        """

        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)

    def collator(self, dataBatch):
    
        if not self.modal == "VO":
            aud_seq_list = [data[0][0] for data in dataBatch]
            aud_padding_mask = torch.zeros((len(aud_seq_list), len(max(aud_seq_list, key=len))), dtype=torch.bool)
            for i, seq in enumerate(aud_seq_list):
                aud_padding_mask[i, len(seq):] = True
            aud_seq_list = pad_sequence(aud_seq_list, batch_first=True)  #可以通过设置 batch_first=True 参数来指定输出的tensor中是否将batch维度放在第一维度
        else:
            aud_seq_list = None
            aud_padding_mask = None
        # visual & len
        if not self.modal == "AO":
            vis_seq_list = pad_sequence([data[0][1] for data in dataBatch], batch_first=True)  #(4,147,1,112,112)   #pad_sequence((none,62,1,112,112))
            vis_len = torch.tensor([len(data[0][1]) for data in dataBatch]) #就是这四个句子每一个的长度 tensor([ 62,  62,  97, 147])   #时间帧上pad
        else:
            vis_seq_list = None
            vis_len = None

        inputBatch = (aud_seq_list, aud_padding_mask, vis_seq_list, vis_len)

        targetinBatch = pad_sequence([data[1] for data in dataBatch], batch_first=True)
        targetoutBatch = pad_sequence([data[2] for data in dataBatch], batch_first=True)
        targetLenBatch = torch.stack([data[3] for data in dataBatch])
 
        #return inputBatch, targetinBatch, targetoutBatch, targetLenBatch   #这里是真的batch那一步  额到这里还不够捏
        # if self.modal == "AO":
        #     inputBatch = (inputBatch[0].float().to('cuda:0'), inputBatch[1].to('cuda:0'), None, None)
        # elif self.modal == "VO":
        #     inputBatch = (None, None, inputBatch[2].float().to('cuda:0'), inputBatch[3].int().to('cuda:0'))
        # else:
        #     inputBatch = (inputBatch[0].float().to('cuda:0'), inputBatch[1].to('cuda:0'), inputBatch[2].float().to('cuda:0'), inputBatch[3].int().to('cuda:0'))
        if self.modal == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.modal == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())

        targetinBatch = targetinBatch.int()
        targetoutBatch = targetoutBatch.int()
        targetLenBatch = targetLenBatch.int()
        targetMask = torch.zeros_like(targetoutBatch, device=targetoutBatch.device)
        targetMask[(torch.arange(targetMask.shape[0]), targetLenBatch.long() - 1)] = 1
        targetMask = (1 - targetMask.flip([-1]).cumsum(-1).flip([-1])).bool()
        concatTargetoutBatch = targetoutBatch[~targetMask]  #(183,)

        return {
            "inputBatch0": inputBatch[0],
            "inputBatch1": inputBatch[1],
            "inputBatch2": inputBatch[2],
            "inputBatch3": inputBatch[3],
            #"inputBatch":inputBatch,
            "targetinBatch": targetinBatch,
            "targetLenBatch": targetLenBatch.long(),
            #"targetinBatch": targetinBatch.to('cuda:0'),
            #"targetLenBatch": targetLenBatch.long().to('cuda:0'),
            'maskw2v': True,
        }     

    def prepare_pretrain_input(self,index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR, numWordsRange, maxLength):  #(3,21)  160
        """
        Function to convert the data sample in the pretrain dataset into appropriate tensors.
        """

        try:
            with open(targetFile, "r") as f:
                lines = f.readlines()
        except:
            print("error")
            print(targetFile)
            print(index)
            return 0, 0, 0, 0

        lines = [line.strip() for line in lines]

        trgt = lines[0][7:]

        coun = trgt.count("{")
        for i in range(coun):
            left = trgt.find("{")
            if left != -1:
                right = trgt.find("}")
                trgt = trgt.replace(trgt[left:right + 2], "")

        trgt=trgt.strip()
        words = trgt.split(" ")

        numWords = len(words) // 3
        if numWords < numWordsRange[0]:   #3   #（numwordsRange 是个tuple（3，21）
            numWords = numWordsRange[0]
        elif numWords > numWordsRange[1]:  #21
            numWords = numWordsRange[1]

        while True:
            # if number of words in target is less than the required number of words, consider the whole target
            if len(words) <= numWords:
                trgtNWord = trgt

                # audio file
                if not modal == "VO":
                    audInp = np.array(h5["flac"][index])
                    audInp = (audInp - audInp.mean()) / audInp.std()
                    if noise is not None:
                        pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                        noise = noise[pos:pos + len(audInp)]
                        noise = noise / np.max(np.abs(noise))
                        gain = 10 ** (noiseSNR / 10)
                        noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                        audInp = audInp + noise
                    audInp = torch.from_numpy(audInp)
                else:
                    audInp = None

                # visual file
                if not modal == "AO":
                    try:
                        vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
                        vidInp = torch.tensor(vidInp).unsqueeze(1)
                        vidInp = transform(vidInp)
                    except:
                        print("error")
                        print(targetFile)
                        print(index)
                        return 0,0,0,0
                else:
                    vidInp = None
            else:
                # make a list of all possible sub-sequences with required number of words in the target
                nWords = [" ".join(words[i:i + numWords])
                        for i in range(len(words) - numWords + 1)]
                nWordLens = np.array(
                    [len(nWord) + 1 for nWord in nWords]).astype(float)

                # choose the sub-sequence for target according to a softmax distribution of the lengths
                # this way longer sub-sequences (which are more diverse) are selected more often while
                # the shorter sub-sequences (which appear more frequently) are not entirely missed out
                ix = np.random.choice(np.arange(len(nWordLens)), p=nWordLens / nWordLens.sum())
                trgtNWord = nWords[ix]

                # reading the start and end times in the video corresponding to the selected sub-sequence
                startTime = float(lines[4 + ix].split(" ")[1])
                endTime = float(lines[4 + ix + numWords - 1].split(" ")[2])

                # audio file
                if not modal == "VO":
                    samplerate = 16000
                    audInp = np.array(h5["flac"][index])  #（81920，）
                    audInp = (audInp - audInp.mean()) / audInp.std()
                    if noise is not None:
                        pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                        noise = noise[pos:pos + len(audInp)]
                        noise = noise / np.max(np.abs(noise))
                        gain = 10 ** (noiseSNR / 10)
                        noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                        audInp = audInp + noise
                    audInp = torch.from_numpy(audInp)
                    audInp = audInp[int(samplerate * startTime):int(samplerate * endTime)]  #！！！！！！！
                else:
                    audInp = None

                # visual file
                if not modal == "AO":
                    videoFPS = 25
                    try:
                        vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]  ##这一句报错x
                        vidInp = torch.tensor(vidInp).unsqueeze(1)
                        vidInp = transform(vidInp)
                        vidInp = vidInp[int(np.floor(videoFPS * startTime)): int(np.ceil(videoFPS * endTime))]
                    except:
                        print("error")
                        print(targetFile)
                        print(index)
                        return 0, 0, 0, 0

                else:
                    vidInp = None

            trgtin = [charToIx[item] for item in trgtNWord]
            trgtout = [charToIx[item] for item in trgtNWord]
            trgtin.insert(0, charToIx["<EOS>"])
            trgtout.append(charToIx["<EOS>"])
            trgtin = np.array(trgtin)
            trgtout = np.array(trgtout)
            trgtLen = len(trgtout)

            inp = (audInp, vidInp)
            trgtin = torch.from_numpy(trgtin)
            trgtout = torch.from_numpy(trgtout)
            trgtLen = torch.tensor(trgtLen)
            inpLen = len(vidInp) if not self.modal == "AO" else len(audInp) / 640
            if inpLen <= maxLength:   #maxlength:160
                break
            elif inpLen > maxLength + 80:
                numWords -= 2
            else:
                numWords -= 1

        return inp, trgtin, trgtout, trgtLen


    def prepare_main_input(index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR):
        """
        Function to convert the data sample in the main dataset into appropriate tensors.
        """
        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:]  #'SO WE NEED YOU TO HELP US IN OUR REVIVAL CAMPAIGN'  'YOU ARE A HEALER IN A STONE AGE VILLAGE'

            coun = trgt.count("{")
            for i in range(coun):
                left = trgt.find("{")
                if left != -1:
                    right = trgt.find("}")
                    trgt  = trgt .replace(trgt [left:right + 2], "")

        trgtin = [charToIx[item] for item in trgt] #[8, 4, 1, 15, 2, 1, 7, 2, 2, 12, 1, 14, 4, 13, 1, 3, 4, 1, 9, 2, 11,
        trgtin.insert(0, charToIx["<EOS>"])  #[39,8,4,...]
        trgtout = [charToIx[item] for item in trgt]
        trgtout.append(charToIx["<EOS>"])   #[..,39] 在最后面加39
        trgtin = np.array(trgtin)
        trgtout = np.array(trgtout)
        trgtLen = len(trgtout)  #50

        # audio file
        if not modal == "VO":
            audInp = np.array(h5["flac"][index])  # ndarray(22528,)
            audInp = (audInp - audInp.mean()) / audInp.std()
            if noise is not None:
                pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                noise = noise[pos:pos + len(audInp)]
                noise = noise / np.max(np.abs(noise))
                gain = 10 ** (noiseSNR / 10)
                noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                audInp = audInp + noise
            audInp = torch.from_numpy(audInp)
        else:
            audInp = None

        # visual file
        if not modal == "AO":
            vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)  #(120,2040,3)
            vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]  #(17,120,120)
            vidInp = torch.tensor(vidInp).unsqueeze(1)  #(17,1,120,120)
            vidInp = transform(vidInp) #(17,1,112,112)
        else:
            vidInp = None

        inp = (audInp, vidInp)
        trgtin = torch.from_numpy(trgtin)
        trgtout = torch.from_numpy(trgtout)
        trgtLen = torch.tensor(trgtLen)

        return inp, trgtin, trgtout, trgtLen





def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = AVSRDataset(dataset_config, tokenizer, split)

    return dataset




class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, device='cpu'):
        self.size = size
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        margin = tensor.shape[-1] - self.size
        hcrop = random.randint(0, margin - 1)
        wcrop = random.randint(0, margin - 1)
        tensor = tensor[:, :, hcrop:-(margin - hcrop), wcrop:-(margin - wcrop)]
        return tensor


class CenterCrop:

    def __init__(self, size, device='cpu'):
        self.size = size
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        crop = (tensor.shape[-1] - self.size) // 2
        tensor = tensor[:, :, crop:-crop, crop:-crop]
        return tensor


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        if random.random() < self.p:
            tensor = torch.flip(tensor, dims=(3,))
        return tensor