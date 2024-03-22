import os
import argparse

def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('path', type=str, help='The path to the image file or folder.')
    args = parser.parse_args()

    if os.path.isdir(args.path):

        for dirpath, dirnames, filenames in os.walk(args.path):
            for filename in filenames:
                if filename.lower().endswith(('.txt')):


                    file_path = os.path.join(dirpath, filename)

                    root, ext = os.path.splitext(file_path)

                    img_path = root + ".jpg"

                    if os.path.isfile(img_path):

                        print(img_path)
                        os.remove(file_path)
                    
                    # else:

                    #     os.remove(file_path)

if __name__ == "__main__":
    main()