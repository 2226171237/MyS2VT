import json
import os

result_path='./result'

reference_dict={
    "info": {
              "description": "test",
              "url": "https://github.com/2226171237",
              "version": "1.0",
              "year": 2019,
              "contributor": "liyajie",
              "date_created": "2020-01-13"
            },
    "images":[],
    "annotations":[]
}

generator_dict=[]

def test(model,dataloader,args,epoch):
    #model.load_weights(WEIGHT_SAVE_PATH+'/model_59200.h5',by_name=True)
    reference_dict['images']=[]
    reference_dict['annotations']=[]
    generator_dict=[]
    image_id=0
    caption_id=0
    for video_features, captions,videos_name in dataloader.get_test_batch(batch_size=args.batch_size):
        generators=model(video_features)
        for i,gen_caption in enumerate(generators):
            sent=[]
            for ii in gen_caption:
                if ii==dataloader.vacabs.word2idx['<eos>']:
                    break
                if ii==dataloader.vacabs.word2idx['<pad>']:
                    continue
                if ii!=dataloader.vacabs.word2idx['<bos>']:
                    sent.append(dataloader.vacabs.idx2word[ii])

            caption=' '.join(sent)
            imgs={
                "image_id":image_id,
                "caption":caption,
                'id':image_id
            }
            generator_dict.append(imgs)

            images = {
                "license": 1,
                "filename": videos_name[i],
                "id": image_id
            }
            reference_dict['images'].append(images)

            for s in captions[i]:
                caption={
                    "image_id":image_id,
                    "id":caption_id,
                    "caption":s
                }
                reference_dict['annotations'].append(caption)
                caption_id+=1
            image_id+=1

    #reference_json_path = os.path.join(result_path, 'reference_%d.json' % epoch)
    generator_json_path = os.path.join(result_path, 'generator_%d.json' % epoch)

    generator_json=json.dumps(generator_dict)
    with open(generator_json_path,'w') as f:
        f.write(generator_json)