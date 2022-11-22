# Cryptography Course
# 9th Semester 2021-2022
# Code for Homework #1

# Exercise 1.2

def eve(Text, Key):
    charText  = [x for x in Text]
    charKey   = [y for y in Key]
    keyLength = len(charKey)

    for k in range(26):
        i = 0
        for t in charText:
            if t.isalpha():
                if ord(t) >= ord('a'):
                    print(chr(((ord(t.lower()) - ord(charKey[i%keyLength]))%26 + ord('a'))), end="")
                else:
                    print(chr(((ord(t.lower()) - ord(charKey[i%keyLength]))%26 + ord('A'))), end="")
                i = i + 1
            else:
                print(t, end="")
        for x in range(keyLength):
            charKey[x] = chr(((ord(charKey[x]) + 1) - ord('a'))%26 + ord('a'))
        print("")
        print("")

# Main
if __name__ == "__main__":
    print("Exercise 1.2")
    eve("Nd Dhy. A dcmgv yk ccob xsieewa svptdwn os ptp Kqg, url gz wazwry vaffu jj t mgzogk tsi os xyextrm lmb hildcmzu. B plsgp plpz oq npw dci Otikigkb usklxc. Egi ahr lrdrd zh g rcr qg wvox zwx hglpsqzw bxrunubydo os wpextrm cgb cik?","cryptography")
