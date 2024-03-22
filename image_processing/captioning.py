import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Models:

    def __init__(self, args):
        
        self.sd_version = args.sd_version

        if args.interrogate_clip:

            self.load_interrogator()

        if args.blip_caption or args.blip_question or args.blip_folder_sort:

            from lavis.models import load_model_and_preprocess

        if args.blip_caption:

            # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
            # this also loads the associated image processors
            self.blip_model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)

        elif args.blip_question or args.blip_folder_sort:

            self.blip_model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=DEVICE)

    def load_interrogator(self):

         # should be used from different conda env because of conflicts with BLIP / LAVIS
        from clip_interrogator import Config, Interrogator

        if self.sd_version == 1:
            self.ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        elif self.sd_version == 2:
            self.ci = Interrogator(Config(clip_model_name="ViT-H-14/laion2b_s32b_b79k"))


def caption_image(image, model, vis_processors):

    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](image).unsqueeze(0).to(DEVICE)
    # generate caption
    caption = model.generate({"image": image})

    print(f"caption: {caption}")

    return caption

def sort_into_folders(image, model, vis_processors, txt_processors):

    questions = [
        'Is this a photo of a room?',
        'Is there a frame around the painting?',
        'Is there a wall visible in the image?',
    ]

    patterns_room = [
        ['yes', 'no', 'no'],
        ['yes', 'yes', 'no'],
        ['yes', 'yes', 'yes']
    ]

    patterns_wall = [
        ['no', 'yes', 'yes'],
        ['no', 'no', 'yes']
    ]

    image = vis_processors["eval"](image).unsqueeze(0).to(DEVICE)

    answers = []

    for question in questions:

        question = txt_processors["eval"](question)
        answer = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
        print(f"{question} {answer[0]}")
        answers.append(answer[0])

    if any(answers == pattern for pattern in patterns_room):

        return "room" 

    elif any(answers == pattern for pattern in patterns_wall):

        return "wall" 

    else:

        return "none"


def sort_by_questions(image, model, vis_processors, txt_processors):

    questions = ['Is there a painting in this image?', 'How many paintings are in this image?', 'Does the painting cover most of the image?']
    patterns = [['yes', '1', 'yes'], ['yes', 'one', 'yes']]

    # questions = ['Is this a photo of a room?', 'Is this a photo of an art installation?']
    # patterns = [['no', 'no']]
    # IS THIS A PHOTOGRAPH OF A ROOM?
    # questions = ['Is there a painting in this image?' , 'is there something in front of the painting?', 'is the painting hanging on a wall?', 'is the image of the painting slanted?', 'is the image of the painting tilted?']
    # patterns = [['yes', 'no', 'no' , 'no']]
    # 'Does the painting fill up most of the image?' 'Does the image show mostly the painting?', 'Does the painting cover most of the image?', 'can we see a room?' drawing!!!
    # # ask a random question.
    # question = "Is there a painting in this image, and if so, how many?"

    image = vis_processors["eval"](image).unsqueeze(0).to(DEVICE)

    answers = []

    for question in questions:

        question = txt_processors["eval"](question)
        answer = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
        print(f"{question} {answer[0]}")
        answers.append(answer[0])

    if any(answers == pattern for pattern in patterns):

        # print('MATCH')

        return 'yes'
    
    else:
        return 'nahh'
    
def interrogate_clip(image, ci):
  
    print(ci.interrogate(image))


