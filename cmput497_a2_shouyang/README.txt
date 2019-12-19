# Authors
    - Shouyang Zhou (ccid:shouyang)

# How to run
    - Run using command line arguments, "langid: langid -mode -inputfile -outputfile"
        - mode: This should the one of either --unsmoothed --laplace --interpolation
        - inputfile: This should be a text file of filepaths to read and evaluate split on new lines.
        - outputfile: This should be a file name that the script will generate in the local folder.

    - Dependencies:
        - The script reads the contents of "train_filepaths.txt", to train a model per filepath-line in the txt.
        - This is hardcoded in the script, I would suggest just subbing the file if needed.
