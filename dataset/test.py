import json
# train_data = json.load(open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/2022-1-26-sgg-data.json','r'))
import numpy as np
import random

import os

path = os.listdir('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/vqa/new_vqa_training_grounding_split_data_inference_512/')
path = ['/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/vqa/new_vqa_training_grounding_split_data_inference_512/'+x for x in path]
b = []
for x in path:
    b.extend(json.load(open(x,'r')))



grounding_dict = {'vg':  {},\
                  'vqa': {}}

for x in tqdm(b):
    dataset = x['dataset']
    q_id = x['q_id']
    if q_id not in grounding_dict[dataset].keys():
        grounding_dict[dataset][q_id] = [{'question_id':x['q_id'],\
                                          'pre_bbox': x['pre_bbox'],\
                                          'logits': x['logits'],\
                                          'positive_token_index': x['positive_token_index'],\
                                          'image_path': x['image_path'],}]
    else:
        grounding_dict[dataset][q_id].append({'question_id':x['q_id'],\
                                              'pre_bbox': x['pre_bbox'],\
                                              'logits': x['logits'],\
                                              'positive_token_index': x['positive_token_index'],\
                                              'image_path': x['image_path'],})


with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/vqa/2022-2-20-grounding_dict_for_new_vqa_training_grounding_split_data_inference_512.json','w') as f:
    json.dump(grounding_dict, f)



# gqa_unbalanced_all_data = []
# raw_data = json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_balanced_questions.json','r'))

# all_objects_dict = json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/chenqianyu_all_objects_id2info.json','r'))
# for q_id,x in raw_data.items():
#     target = x.copy()
#     question = target['question']
#     data = {}
#     if 'question' in target['annotations']:
#         if len(target['annotations']['question']) > 0:
#             q_ann = target['annotations']['question'].copy()
#             last_token_dict = {}
#             for obj_pointer in q_ann:
#                 if ':' not in obj_pointer:
#                     last_token_dict[obj_pointer] = obj_pointer
#                 else:
#                     tt = str(int(obj_pointer.split(':')[-1]) - 1)
#                     last_token_dict[tt] = obj_pointer
#             question = question.split(' ')
#             new_question = []
#             ann_token_id = []
#             for index, token in enumerate(question):
#                 key = str(index)
#                 if key in last_token_dict:
#                     token = [token, index]
#                     ann_token_id.append([index, q_ann[last_token_dict[key]]])
#                 new_question.append(token)
#             answer = target['answer']
#             ann_token_id = sorted(ann_token_id)
#             bbox_list = []
#             for index, obj_id in ann_token_id:
#                 bbox = [all_objects_dict[obj_id]['x'], all_objects_dict[obj_id]['y'], all_objects_dict[obj_id]['w'], all_objects_dict[obj_id]['h']]
#                 bbox_xyxy = [bbox[0], bbox[1], bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1]
#                 bbox_list.append(bbox_xyxy.copy())
#             data['tokens_positive'] = ann_token_id.copy()
#             data['question'] = new_question.copy()
#             data['bbox_list'] = bbox_list.copy()
#             data['answer'] = answer
#             data['file_name'] = '/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/images/'+target['imageId']+'.jpg'
#             data['not_crop_bbox_list'] = bbox_list.copy()
#             data['no_bbox'] = 0
#             data['question_id'] = q_id
#             gqa_unbalanced_all_data.append(data.copy())
#     #     else:
#     #         data['question'] = target['question'].split(' ')
#     #         data['no_bbox'] = 1
#     #         data['file_name'] = '/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/images/'+target['imageId']+'.jpg'
#     #         data['bbox_list'] = [[0,0,1000,1000]]
#     #         data['not_crop_bbox_list'] = [[0,0,1000,1000]]
#     #         data['answer'] = target['answer']
#     #         data['question_id'] = q_id
#     #         gqa_unbalanced_all_data.append(data.copy())   
#     # else:
#     #     data['question'] = target['question'].split(' ')
#     #     data['no_bbox'] = 1
#     #     data['file_name'] = '/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/images/'+target['imageId']+'.jpg'
#     #     data['bbox_list'] = [[0,0,1000,1000]]
#     #     data['not_crop_bbox_list'] = [[0,0,1000,1000]]
#     #     data['answer'] = target['answer']
#     #     data['question_id'] = q_id
#     #     gqa_unbalanced_all_data.append(data.copy())

# for x in gqa_unbalanced_all_data:
#     if x['no_bbox'] == 0:
#         for m,n in x.items():
#             print(m,n)
#         break

# # with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-15-gqa-val-grounding-balanced-single-bbox-data.json','w') as f:
# #     json.dump(gqa_unbalanced_all_data,f)

# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-15-gqa-train-balanced-grounding-all-bbox-data.json','w') as f:
#     json.dump(gqa_unbalanced_all_data, f)


# raw_data = json.load(open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-15-gqa-train-balanced-grounding-all-bbox-data.json','r'))
# temp_dict = {}
# for ann in raw_data:
#     num = 0
#     question = ann['question']
#     for question_token in question:
#         if isinstance(question_token, list):
#             num+=1
#     q_id = ann['question_id']
#     assert q_id not in temp_dict.keys()
#     temp_dict[q_id] = []
#     for index in range(num):
#         ann_single = ann.copy()
#         ann_single['bbox_choice'] = index
#         temp_dict[q_id].append(ann_single.copy())

# gqa_balanced_grounding_single_bbox_data = []
# for q_id, ann_list in temp_dict.items():
#     for x in ann_list:
#         bbox_choice = x['bbox_choice']
#         new_question = []
#         s_question = x['question']
#         num=0
#         for question_token in s_question:
#             if isinstance(question_token, list):
#                 if num == bbox_choice:
#                     new_question.append(question_token)
#                     num +=1 
#                 else:
#                     new_question.append(question_token[0])
#             else:
#                 new_question.append(question_token)
#         target = x.copy()
#         target['question'] = new_question
#         gqa_balanced_grounding_single_bbox_data.append(target.copy())

# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-15-gqa-train-balanced-grounding-single-bbox-data.json','w') as f:
#     json.dump(gqa_balanced_grounding_single_bbox_data, f)






# a = json.load(open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-7-gqa-balanced-finetune-data.json','r'))
# b=[]
# for x in a:
#     s = x.copy()
#     s['dataset'] = 'vqa'
#     s['image'] = x['file_name']
#     s['question'] = x['normal_question']
#     s['answer'] = [x['answer']]
#     b.append(s.copy())



# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-14-gqa-balanced-for-albef-vqa-data.json','w') as f:
#     json.dump(b,f)
# gqa_unbalanced_all_data = []
# raw_data = {}
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_0.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_1.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_2.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_3.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_4.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_5.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_6.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_7.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_8.json','r')))
# raw_data.update(json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/train_all_questions/train_all_questions_9.json','r')))


# all_objects_dict = json.load(open('/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/chenqianyu_gqa_objects_id2info.json','r'))
# for q_id,x in raw_data.items():
#     target = x.copy()
#     question = target['question']
#     data = {}
#     if 'question' in target['annotations']:
#         if len(target['annotations']['question']) > 0:
#             q_ann = target['annotations']['question'].copy()
#             last_token_dict = {}
#             for obj_pointer in q_ann:
#                 if ':' not in obj_pointer:
#                     last_token_dict[obj_pointer] = obj_pointer
#                 else:
#                     tt = str(int(obj_pointer.split(':')[-1]) - 1)
#                     last_token_dict[tt] = obj_pointer
#             question = question.split(' ')
#             new_question = []
#             ann_token_id = []
#             for index, token in enumerate(question):
#                 key = str(index)
#                 if key in last_token_dict:
#                     token = [token, index]
#                     ann_token_id.append([index, q_ann[last_token_dict[key]]])
#                 new_question.append(token)
#             answer = target['answer']
#             ann_token_id = sorted(ann_token_id)
#             bbox_list = []
#             for index, obj_id in ann_token_id:
#                 bbox = [all_objects_dict[obj_id]['x'], all_objects_dict[obj_id]['y'], all_objects_dict[obj_id]['w'], all_objects_dict[obj_id]['h']]
#                 bbox_xyxy = [bbox[0], bbox[1], bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1]
#                 bbox_list.append(bbox_xyxy.copy())
#             data['tokens_positive'] = ann_token_id.copy()
#             data['question'] = new_question.copy()
#             data['bbox_list'] = bbox_list.copy()
#             data['answer'] = answer
#             data['file_name'] = '/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/images/'+target['imageId']+'.jpg'
#             data['not_crop_bbox_list'] = bbox_list.copy()
#             data['no_bbox'] = 0
#             data['question_id'] = q_id
#             gqa_unbalanced_all_data.append(data.copy())
#         else:
#             data['question'] = target['question'].split(' ')
#             data['no_bbox'] = 1
#             data['file_name'] = '/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/images/'+target['imageId']+'.jpg'
#             data['bbox_list'] = [[0,0,1000,1000]]
#             data['not_crop_bbox_list'] = [[0,0,1000,1000]]
#             data['answer'] = target['answer']
#             data['question_id'] = q_id
#             gqa_unbalanced_all_data.append(data.copy())   
#     else:
#         data['question'] = target['question'].split(' ')
#         data['no_bbox'] = 1
#         data['file_name'] = '/mnt/sfs_turbo/chenqianyu/ALBEF_Datasets/gqa/images/'+target['imageId']+'.jpg'
#         data['bbox_list'] = [[0,0,1000,1000]]
#         data['not_crop_bbox_list'] = [[0,0,1000,1000]]
#         data['answer'] = target['answer']
#         data['question_id'] = q_id
#         gqa_unbalanced_all_data.append(data.copy())

# for x in gqa_unbalanced_all_data:
#     if x['no_bbox'] == 0:
#         for m,n in x.items():
#             print(m,n)
#         break

# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-14-gqa-unbalanced-all-split-new-right-grounding-info-data.json','w') as f:
#     json.dump(gqa_unbalanced_all_data,f)

# a = json.load(open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-6-gqa-all-split-data.json','r'))
# b=[]
# for x in a:
#     s = x.copy()
#     s['dataset'] = 'vqa'
#     s['image'] = x['file_name']
#     s['question'] = x['normal_question']
#     s['answer'] = [x['answer']]
#     b.append(s.copy())


# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-14-gqa-unbalanced-all-split-for-albef-vqa-data.json','w') as f:
#     json.dump(b,f)
# a = json.load(open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-7-gqa-balanced-finetune-data.json','r'))
# b =[]
# for x in a:
#     if x['no_bbox']==0:
#         b.append(x.copy())


# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/gqa/2022-2-8-gqa-balanced-finetune-grounding-data.json','w') as f:
#     json.dump(b,f)





# split_train_data = []
# for x in train_data:
#     sgg_dict = x.copy()
#     for rel_pair in sgg_dict['rel_tri']:
#         split_sgg_dict = sgg_dict.copy()
#         split_sgg_dict['rel_tri'] = rel_pair
#         split_sgg_dict['with_rel'] = 1
#         split_train_data.append(split_sgg_dict)
#     neg_rel_pair_list = []
#     for rel_pair in sgg_dict['rel_tri']:
#         # split_sgg_dict = sgg_dict.copy()
#         neg_rel_pair_list.append(str(rel_pair[1])+'_'+str(rel_pair[0]))
#     # split_sgg_dict = sgg_dict.copy()
#     # split_sgg_dict['rel_tri'] = neg_rel_pair_list
#     # split_sgg_dict['with_rel'] = 1
#     # split_train_data.append(split_sgg_dict)
#     neg_rel_pair_list.extend(sgg_dict['un_rel_pair'])
#     a = [m for m in range(len(neg_rel_pair_list))]
#     a = np.array(a)
#     # random.shuffle(neg_rel_pair_list)
#     if len(x['rel_tri']) * 3 < len(neg_rel_pair_list):
#         split_num = len(x['rel_tri'])*3
#         a = np.array_split(a, split_num)
#         for split_index_list in a:
#             split_neg_pair = x.copy()
#             split_neg_rel_pair_list = [neg_rel_pair_list[index] for index in split_index_list]
#             split_neg_pair['un_rel_pair'] = split_neg_rel_pair_list
#             split_neg_pair['with_rel'] = 0
#             split_train_data.append(split_neg_pair)
#     else:
#         split_neg_pair = x.copy()
#         split_neg_pair['un_rel_pair'] = neg_rel_pair_list
#         split_neg_pair['with_rel'] = 0
#         split_train_data.append(split_neg_pair)



# t = []
# for x in split_train_data:
#     if x['with_rel'] == 1:
#         if len(x['rel_tri'])>0:
#             t.append(x)
#     elif x['with_rel'] == 0:
#         if len(x['un_rel_pair'])>0:
#             t.append(x)


# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/sgg_data/2022-2-6-sgg-train-data-split-with-BA-TRI.json','w') as f:
#     json.dump(t, f)


# file_list = [
#               '/mnt/sfs_turbo/chenqianyu/albef_anno_data/albef_all_data/ALBEF_512/new_crop_bbox_corpus/flickr_train_corp_bbox_corpus.json',\
#              '/mnt/sfs_turbo/chenqianyu/albef_anno_data/albef_all_data/ALBEF_512/new_crop_bbox_corpus/gqa_train_blanced_corp_bbox_corpus.json',\
#              '/mnt/sfs_turbo/chenqianyu/albef_anno_data/albef_all_data/ALBEF_512/new_crop_bbox_corpus/new_refcocop_train_with_vinvl_bbox.json',\
#               '/mnt/sfs_turbo/chenqianyu/albef_anno_data/albef_all_data/ALBEF_512/new_crop_bbox_corpus/new_refcoco_train_with_vinvl_bbox.json',\
#               '/mnt/sfs_turbo/chenqianyu/albef_anno_data/albef_all_data/ALBEF_512/new_crop_bbox_corpus/new_refcocog_train_with_vinvl_bbox.json',\
#               '/mnt/sfs_turbo/chenqianyu/albef_anno_data/albef_all_data/ALBEF_512/new_crop_bbox_corpus/chenqianyu_albef_new_vg_region_seprated_crop_bbox_corpus.json',\
#             ]
# a = []
# for x in file_list:
#     d = json.load(open(x,'r'))
#     print(d[1]['file_name'])
#     print('\n\n')
# print('\n\n')
# print(len(a))
# print(a[1])
# for x,y in a[1].items():
#     print(x,y)
#     print('\n')
# print('\n\n')
# print(a[2])
# print(a[1])


# vcr_pretrain_data = []
# for x in a:
#     if 'with_answer' in x:
#         ann = x.copy()
#         for m,n in enumerate(ann['answer_choices']):
#             target = ann.copy()
#             if m != ann['answer_label']:
#                 target['wrong_answer'] = n
#                 target['label'] = 0
#             else:
#                 target['right_answer'] = n
#                 target['label'] = 1
#             vcr_pretrain_data.append(target)
#     elif 'with_rationale' in x:
#         ann = x.copy()
#         for m,n in enumerate(ann['rationale_choices']):
#             target = ann.copy()
#             if m != ann['rationale_label']:
#                 target['right_answer'] = ann['right_answer']
#                 target['wrong_rationale'] = n
#                 target['label'] = 0
#             else:
#                 target['right_answer'] = ann['right_answer']
#                 target['right_rationale'] = n
#                 target['label'] = 1
#             vcr_pretrain_data.append(target)


# print(len(vcr_pretrain_data))
# with open('/mnt/sfs_turbo/chenqianyu/albef_downstream_tasks/2022_01_06_vcr_pretrain_prompt_data.json','w') as f:
#     json.dump(vcr_pretrain_data, f)
# a = vcr_pretrain_data
# print('\n\n')
# for x ,y in a[0].items():
#     print(x,y)
# print('\n\n')
# for x,y in a[4].items():
#     print(x,y)
# print('\n\n')
# for x ,y in a[1].items():
#     print(x,y)
# print('\n\n')
# for x,y in a[5].items():
#     print(x,y)
# print('\n\n')