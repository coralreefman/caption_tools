import os
import json
from PIL import Image
from shutil import copyfile
from . import operations
from . import face_detection
from . import captioning

class ImageProcessor:

    def __init__(self, path, out_path, models, args):

        self.path = path
        self.out_path = out_path

        self.format = args.format
        self.quality = args.quality

        self.face_detection_threshold = args.face_detection_threshold
        self.sd_version = args.sd_version

        self.models = models

        self.min_size = args.min_size
    
    def open_image(self):

        self.image = Image.open(self.path).convert('RGB')
        self.width, self.height = self.image.size
         
    def is_min_size(self):

        if self.width > self.min_size or self.height > self.min_size:
            return True
        else:
            return False
 
    def crop_by_percentage(self, amount):
        
        self.image = operations.crop_by_percentage(self.image, amount)

    def crop_from_background(self):

        self.image = operations.crop_from_background(self.image)

    def face_detection_square(self):

        self.faces = face_detection.detect_faces(self.image, self.face_detection_threshold)

        self.image = face_detection.draw_squares(self.image, self.faces)

    def face_detection_blur(self):

        self.faces = face_detection.detect_faces(self.image, self.face_detection_threshold)

        self.image = face_detection.blur_faces(self.image, self.faces)

    def save_caption(self):

        if not os.path.exists(os.path.dirname(self.out_path)):

            os.mkdir(os.path.dirname(self.out_path))  

        caption_path = f'{os.path.splitext(self.out_path)[0]}.txt'

        # file_exists = os.path.isfile(caption_path)
        # if not file_exists:

        with open(caption_path, 'a') as f:

            # if not file_exists:
            #     prepend = ''
            # else:
            #     if f.tell() == 0:
            #         prepend = ''
            #     else:
            #         prepend = ', '
            prepend = ''

            if hasattr(self, 'caption_blip'):
                # if self.caption_blip[0] not in f:
                f.write(f'{prepend}{self.caption_blip[0]}, ')
            if hasattr(self, 'caption_clip'):
                f.write(f'{prepend}{self.caption_clip}, ')
            if hasattr(self, 'caption_metadata'):
                # if self.caption_metadata not in f:
                f.write(f'{prepend}{self.caption_metadata}, ')

    def metadata_caption(self):

        metadata_path = os.path.join(os.path.dirname(self.path), 'metadata.json')

        if os.path.exists(metadata_path):
            
            f = open(metadata_path, 'r') 

            data = json.load(f)

            if not data['group_show'] and len(data['artist']) > 1:

                self.caption_metadata = f"in the style of {data['artist']}"
                print(self.caption_metadata) 

    def blip_caption(self):

        self.caption_blip = captioning.caption_image(self.image, self.models.blip_model, self.models.vis_processors)

    def blip_sort_by_questions(self):

        self.answer = captioning.sort_by_questions(self.image, self.models.blip_model, self.models.vis_processors, self.models.txt_processors)

    def blip_sort_into_folders(self):

        self.answer = captioning.sort_into_folders(self.image, self.models.blip_model, self.models.vis_processors, self.models.txt_processors)

    def interrogate_clip(self):

        self.caption_clip = captioning.interrogate_clip(self.image, self.models.ci)

    def convert(self, mode):

        # mode: The mode to convert to. Must be a string like 'RGB', 'RGBA', 'L', etc.
        self.image = self.image.convert(mode)

    def copy_metadata(self):

        metadata_in = os.path.join(os.path.dirname(self.path), 'metadata.json')
        metadata_out = os.path.join(os.path.dirname(self.out_path), 'metadata.json')

        if os.path.exists(os.path.dirname(self.out_path)) and os.path.isfile(metadata_in) and not os.path.isfile(metadata_out):
            copyfile(metadata_in, metadata_out)

    def copy_captions(self):

        root_in, ext_in = os.path.splitext(self.path)
        root_out, ext_out = os.path.splitext(self.out_path)  

        captions_in = root_in + '.txt'
        captions_out = root_out + '.txt'

        print(captions_in, captions_out)

        if os.path.isfile(captions_in) and not os.path.isfile(captions_out):
            copyfile(captions_in, captions_out)

    def save(self):
        """
        Save the image to a file.

        Args:
            path: The path to save the image. If not specified, overwrites the original image.
            format: The format to use for the saved image. Defaults to 'JPEG'.
            quality: The quality to use for the saved image (only applicable for some formats). Defaults to 100.
        """
        
        # if the image wasn't opened, e.g. in text only mode, skip saving
        if not hasattr(self, 'image'):

            pass 

        else:
            # If the image mode is not 'RGB' and the output format is 'JPEG', convert the image to 'RGB' mode
            if self.image.mode != 'RGB' and format.upper() == 'JPEG':

                self.image = self.image.convert('RGB')

            # overwrite in place if output_dir is not given
            if self.out_path is None:

                self.out_path = self.path
            
            if not os.path.exists(os.path.dirname(self.out_path)):

                os.mkdir(os.path.dirname(self.out_path))      

            self.image.save(self.out_path, format=self.format, quality=self.quality)
