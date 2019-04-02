import tensorflow as tf
import numpy as np
from bitstring import BitArray
import matplotlib.pyplot as plt

class Image(object):
    def __init__(self,file_name):
        self.f = open(file_name,"rb")
        self.file_name = file_name
        self.DGR_HEADER={}
        self.DOC_IMG = {}
        #self.DOC_IMG_data = 

        self.ReaddgrFile2Img()

    def ReaddgrFile2Img(self):
        #self.DGR_HEADER = {}
        #read the head information of the dgr file
        self.DGR_HEADER['HdSize'] = self.f.read(4)
        hdSize = int.from_bytes(self.DGR_HEADER['HdSize'],byteorder='little')
        self.DGR_HEADER['FmCode'] = self.f.read(8)
        self.DGR_HEADER['Illustration'] = self.f.read(hdSize-36)
        self.DGR_HEADER['CodeType'] = self.f.read(20)
        self.DGR_HEADER['CodeLen'] = self.f.read(2)
        self.DGR_HEADER['BitsperPix'] = self.f.read(2)
        bitsppix = int.from_bytes(self.DGR_HEADER['BitsperPix'],byteorder='little')

        #the annotation information of document image
        #read the height and width of the document image

        self.DOC_IMG['ImgHei'] = self.f.read(4)
        self.DOC_IMG['ImgWid'] = self.f.read(4)
        self.DOC_IMG['LineNum'] = self.f.read(4)
        imgHei = int.from_bytes(self.DOC_IMG['ImgHei'],byteorder='little')
        imgWid = int.from_bytes(self.DOC_IMG['ImgWid'],byteorder='little')

        #the lineInfo list
        LINE_INFO_list = []
        line_num = int.from_bytes(self.DOC_IMG['LineNum'],byteorder='little')
        mapLineList = []

        for i in range(line_num):
            #the annotation information of a text line
            LINE_INFO = {}
            LINE_INFO['WordNum'] = self.f.read(4)
            #the wordInfo list
            WORD_INFO_list = []
            word_num = int.from_bytes(LINE_INFO['WordNum'],byteorder='little')
            print("word_num:",word_num)

            fpos = self.f.tell()
            topMin = 99999
            bottomMax = 0
            leftMin = 0
            rightMax = 0
            preRight = 0
            sumMargin = [0 for i in range(word_num)]

            for word_j in range(word_num):

                # the annotation information of a word
                WORD_INFO = {}
                WORD_INFO['Label'] = self.f.read(int.from_bytes(self.DGR_HEADER['CodeLen'],byteorder = 'little'))
                WORD_INFO['Top'] = self.f.read(2)
                WORD_INFO['Left'] = self.f.read(2)
                WORD_INFO['Height'] = self.f.read(2)
                WORD_INFO['Width'] = self.f.read(2)
                 
                tmp_top = int.from_bytes(WORD_INFO['Top'],byteorder='little')+1
                tmp_left = int.from_bytes(WORD_INFO['Left'],byteorder='little')+1
                tmp_hei = int.from_bytes(WORD_INFO['Height'],byteorder='little')
                tmp_wid = int.from_bytes(WORD_INFO['Width'],byteorder='little')

                # print("top",tmp_top,"left:",tmp_left,
                #         "hei:",tmp_hei,"wid:",tmp_wid)

                if topMin > tmp_top:
                    topMin = tmp_top
                if tmp_left+tmp_wid-1>rightMax :
                    rightMax = tmp_left+tmp_wid-1
                if tmp_hei+tmp_top-1>bottomMax :
                    bottomMax = tmp_hei+tmp_top-1

                if bitsppix == 1:
                    byteHori = int((tmp_wid+7)/8)
                    self.f.seek(byteHori*tmp_hei,1)
                elif bitsppix == 8:
                    self.f.seek(tmp_wid*tmp_hei,1)

                if word_j == 0:
                    leftMin = tmp_left
                
            self.f.seek(fpos,0)

            #tmp_data = [255 for i in range((bottomMax-topMin+1)*(rightMax-leftMin+1))]
            DOC_IMG_line = np.ones((bottomMax-topMin+1,rightMax-leftMin+1),dtype=int)
            #print(DOC_IMG_line.shape)

            for word_j in range(word_num):

                WORD_INFO = {}
                WORD_INFO['Label'] = self.f.read(int.from_bytes(self.DGR_HEADER['CodeLen'],byteorder = 'little'))
                WORD_INFO['Top'] = self.f.read(2)
                WORD_INFO['Left'] = self.f.read(2)
                WORD_INFO['Height'] = self.f.read(2)
                WORD_INFO['Width'] = self.f.read(2)
                WORD_INFO_list.append(WORD_INFO)
                 
                tmp_top = int.from_bytes(WORD_INFO['Top'],byteorder='little')+1
                tmp_left = int.from_bytes(WORD_INFO['Left'],byteorder='little')+1
                tmp_hei = int.from_bytes(WORD_INFO['Height'],byteorder='little')
                tmp_wid = int.from_bytes(WORD_INFO['Width'],byteorder='little')


                if bitsppix == 1:
                    byteHori = int((tmp_wid+7)/8)
                    #tmp_data = [255 for i in range(tmp_hei*(byteHori*8))]
                    tmp_data = np.ones((tmp_hei,byteHori*8),dtype= int)

                    for m in range(tmp_hei):
                        for n in range (byteHori):
                            byte = self.f.read(1)
                            bit_array = BitArray(byte)
                            for i_bit in range(8):
                                tmp_data[m,(n)*8+i_bit] = bit_array[7-i_bit]
                    data = tmp_data[:,:tmp_wid]

                    for m in range(tmp_hei):
                        for n in range(tmp_wid):
                            DOC_IMG_line[tmp_top-topMin+m,tmp_left - leftMin+n] = DOC_IMG_line[tmp_top-topMin+m,tmp_left - leftMin+n] and (not(data[m,n]))

                elif bitsppix == 8:
                    for m in range(tmp_hei):
                        for n in range(tmp_wid):
                            DOC_IMG_line[tmp_top-topMin+m,tmp_left-leftMin+n] = self.f.read(1)

            mapLineList.append(DOC_IMG_line)
            LINE_INFO['WordInfo'] = WORD_INFO_list
            #print(LINE_INFO['WordInfo'])  
            LINE_INFO_list.append(LINE_INFO)
        self.DOC_IMG['LineInfo'] = LINE_INFO_list
        self.DOC_IMG['mapLine'] = mapLineList

    def get_lineNum(self):
        return int.from_bytes(self.DOC_IMG['LineNum'],byteorder='little')

    def get_lineData(self,lineInd):
        line_label= []
        word_num = int.from_bytes(self.DOC_IMG['LineInfo'][lineInd]['WordNum'],byteorder='little')
        for i in range(word_num):
            line_label.append(self.DOC_IMG['LineInfo'][lineInd]['WordInfo'][i]['Label'])
        line_data = self.DOC_IMG['mapLine'][lineInd]
        return line_label,line_data

    def priImg(self,data):
        plt.figure()
        plt.imshow(data)
        plt.show()

if __name__ == '__main__':
    temp_image = Image(r'/home/zhang/桌面/HIT-MW database/02 HIT-MW GT (binary)/黑白格式/b04010201.dgr')