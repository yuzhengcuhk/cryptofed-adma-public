import os
import subprocess

with open('ABsum.txt', 'w') as sumfile:
    with open('alicegenpara.txt', 'r') as alicefile:
        with open('bobgenpara_rand.txt', 'r') as bobfile:
            alicedata = alicefile.readline()
            bobdata = bobfile.readline()
            while alicedata:
                aodom = alicedata.split()
                bodom = bobdata.split()
                anum = map(float, aodom)
                bnum = map(float, bodom)
                alst = (list(anum))
                blst = (list(bnum))
                Alen = len(alst)
                Blen = len(blst)
                for i in range(0, Alen):
                    blst[i] = alst[i]+blst[i]
                    sumfile.write(str(blst[i]))
                    sumfile.write(' ')
                sumfile.write("\n")
                alicedata = alicefile.readline()
                bobdata = bobfile.readline()
            '''for aline, bline in alicedata, bobdata:
                aodom = aline.split()
                bodom = bline.split()
                anum = map(float, aodom)
                bnum = map(float, bodom)
                alst = (list(anum))
                blst = (list(bnum))
                Alen = len(alst)
                Blen = len(blst)
                for i in range(0, Alen):
                    blst[i] = alst[i]+blst[i]
                    sumfile.write(str(blst[i]))
                    sumfile.write(' ')
                sumfile.write("\n")
#同态加密
#subprocess.call([os.getcwd()+'/alice_puben'])'''

