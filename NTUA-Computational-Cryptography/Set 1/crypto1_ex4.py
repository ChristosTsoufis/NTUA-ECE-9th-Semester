# Cryptography Course
# 9th Semester 2021-2022
# Code for Homework #1

# Exercise 4

# the following function calculates the IC in a given text or array

def coincidenceIndex(Text):
    length = len(Text)
    result = 0
    # frequency table
    freq = [0]*26

    for i in Text:
        freq[ord(i) - ord('A')] = freq[ord(i) - ord('A')] + 1
    for i in freq:
        # probability calculation
        result = result + (i*(i-1)) / (length*(length-1)) 
    
    return result


# the following function calculates Vigenere cipher
# input: ciphertext that is a result of Vigenere cipher
# output: 26 different versions of the plaintext and its key

def vigenere(Text):
    #Coincidence Index of original text
    ltext = coincidenceIndex(Text)
    charText = [x for x in Text]
    # array that saves the mean value of IC for different key sizes 
    meanIndexes = []
    # finds possible key size using the formula
    possibleKeyLength = round((0.065-0.038) / (ltext-0.038))
    # the formula is based in probabilities so search for a region around the possible key size
    for i in range(possibleKeyLength*2):
        temp = 0
        for x in range(i+1):
            temp = temp + coincidenceIndex(charText[x::(i+1)])
        # keep the mean value of IC for the different groups
        meanIndexes.append(temp/(i+1))
    
    # takes key size equal to the one with the biggest mean value of IC
    keyLength = meanIndexes.index(max(meanIndexes)) + 1
    shiftList = []
    list1 = charText[1::keyLength]

    # finds the relative shifts with the second group of letters (list1)
    for i in range(keyLength):
        list2 = charText[i::keyLength]
        index = 0
        shift = []

        # finds the IC for all the shifts and keeps the best
        for k in range(26):
            index = coincidenceIndex(list2 + list1)
            shift.append(index)
            
            for x in range(len(list2)):
                # shift by -1 the group of letters
                list2[x] = chr(((ord(list2[x]) - 1) - ord('A'))%26 + ord('A'))

        shiftList.append(shift.index(max(shift)))
    
    key = [0]

    # finds the possible key that satisfies the shifting
    for i in range(keyLength-1):
            key.append((shiftList[i+1] - shiftList[0])%26)
    
    # prints 26 different shiftings of the key and the related text
    for k in range(26):
        # shift the key and print it
        for x in range(keyLength):
            key[x] = (key[x] + 1)%26
            print(chr(key[x] + ord('A')), end="")
        print("")
        print("")

        i = 0
        for t in charText:
            # shift the letters accoding to the key
            print(chr(((ord(t) - key[i%keyLength] - ord('A'))%26 + ord('A'))), end="")
            i = i + 1
        print("")
        print("")
        print("-----------------------------------------------------------------------------------------------------------------------")

# Main
if __name__ == "__main__":
    print("Exercise 4")
    vigenere("KUDLEZSIOGOOSMWJICKIELOLOVTDOECJZYWNCHIOAAKILDVUDWQIPJVKRPVLTLIOZATLJUCSMOIWLCKVBBLNZBJUCSMOIWLCKVURLYLZPZPFCVNDIYJLBENHEMCYWGVPFPAWUVHSUGQWCOBTOSFEPPEKPWLTSZZAOIIVUMCETWUPYOXGZAIONAHZCRNBIOFACMHOBIIVUJMEZPFIIEWWYMVPDAOJWFEWVWHYRQGKBIOTYZCCRWWOIUEVZZGPEYTSWFMPUCMOCBSKGIKCEEPPOZPGUTSWFMPUCFOFULEPPEZEEPPOZCYKIPAMABOYATSMTXAPESSQWCZPFSYSZCWYLXOSLTVENMIYBSPWQZWNYYRZEHNQDRFOFKLTVPDCIDQETWUOCEYQYEWVBKRPIXYGPETAJAEXQWOAOMMWOSFDOEXJFZAFORNZBEUUBULTSMXRQLOFOFNZXTJPLGZMDTWULHCVELXLOYYTLZCOMTGPTSGPYBFBKAJGZAISOAHPJZCAGBVMHPTIPZYBRNPTRLSOUADTTBQOQHRCWHYISOZEYBQUZURAHPICIPFBZEPAENMNKYKFXZTBIOWAEPZLBIOPNQQYOBFKUDSMMKVECFOFETZPISZMTOSZBIKAHTALXZPGZMLGRUAXSMTLVOLISVLTJWFXJFXKIYOTTBIOHRNPPXAIKUDMMQUZHVHDYMDYNPBLVPVLYPFVVVPAENMBBYOHBSGBGVPEDAZNMMYCEDIWYWURLBZEENIUSZSEIMRM")
