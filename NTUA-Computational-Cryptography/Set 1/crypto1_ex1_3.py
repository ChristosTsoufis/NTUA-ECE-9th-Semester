# Cryptography Course
# 9th Semester 2021-2022
# Code for Homework #1

# Exercise 1.3

def bob(Text, Key1, Key2):
    charText  = [x for x in Text]
    charKey   = [y for y in Key1]
    keyLength = len(charKey)

    for x in range(keyLength):
        charKey[x] = chr(((ord(charKey[x]) + Key2) - ord('a'))%26 + ord('a'))
    
    i = 0
    
    for t in charText:
        if t.isalpha():
            if ord(t) >= ord('a'):
                print(chr(((ord(t.lower()) + ord(charKey[i%keyLength]) - 2*ord('a'))%26 + ord('a'))), end="")
            else:
                print(chr(((ord(t.lower()) + ord(charKey[i%keyLength]) - 2*ord('a'))%26 + ord('A'))), end="")
            i = i + 1
        else:
            print(t, end="")
    print("")

# Main
if __name__ == "__main__":
    print("Exercise 1.3")
    bob("Hello there. We have to do something in order to outsmart Eve. However, if we must use the Vigenere cipher, we should probably use the One Time Pad method with a random big key just to be sure.", "cryptography", 4)
