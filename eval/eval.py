import torch
import json
from tqdm import tqdm

@torch.no_grad()
def vcr_validate(model, data_loader, tokenzier, device, setting):
    # model.eval()
    total_count = 0.0
    right_count = 0.0
    for i, (images, text_list, ans_labels, _) in enumerate(data_loader):
        images = images.to(device, non_blocking=True).view(-1,3,512,512)
        text_list_re_ = [[] for x in range(len(text_list[3]))]
        for _, x in enumerate(text_list[:4]):
            for index, y in enumerate(x):
                text_list_re_[index].append(y)
        
        text_input=[]
        for x in text_list_re_:
            text_input.extend(x)
        text = tokenzier(text_input, padding='longest', truncation=True, max_length=512, return_tensors="pt").to(device) 
        assert images.shape[0] == text.input_ids.shape[0]
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        image_embeds = model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
        text_output = model.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                         return_dict = True, mode = 'text') 
        text_embeds = text_output.last_hidden_state
        output_pos = model.text_encoder.bert(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )
        vl_embeddings = output_pos.last_hidden_state[:,0,:]
        vl_output = model.itm_head(vl_embeddings)
        result = vl_output[:,1].view(-1,4)
        choice = result.argmax(dim=1)
        assert len(choice) == len(ans_labels)
        for x,y in zip(choice, ans_labels):
            if x==y:
                right_count+=1
            total_count+=1
            # if i %200 ==0:0

            #     print(right_count/total_count)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print('total num: 26534.0 test result for Q2A in VCR: 71.3%')
    print('+                                                     +')
    print(f" total num: {total_count} test result for {setting} in VCR: {round(right_count/total_count, 3)*100.0}%")
    print('+                                                     +')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')


@torch.no_grad()
def gqa_test(model, data_loader, tokenizer, device, answer_dict_path):
    model.eval()
    test_results = []
    results = []
    all_answer_dict = json.load(open(answer_dict_path))
    gqa_answer_list = []
    for x in range(len(all_answer_dict)):
        gqa_answer_list.append(all_answer_dict[str(x)])
    gqa_answer_input =  tokenizer(gqa_answer_list, padding='longest', return_tensors='pt').to(device).input_ids
    gqa_answer_index_list = []
    for answer in gqa_answer_input:
        gqa_answer_index_list.append(answer[1:])
    gqa_answer_index_list = torch.stack(gqa_answer_index_list, dim=0).to(device)
    first_index = gqa_answer_index_list[:,0]
    second_index = gqa_answer_index_list[:,1]
    third_index = gqa_answer_index_list[:,2]
    fourth_index = gqa_answer_index_list[:,3]
    fifth_index = gqa_answer_index_list[:,4]

    gqa_answer_length_list = []
    for answer in gqa_answer_index_list:
        gqa_answer_length_list.append(1/float(len(torch.where(answer!=0)[0])))
    gqa_answer_length_list = torch.tensor(gqa_answer_length_list)
    for i, (image, text, right_answer_list,q_ids) in enumerate(data_loader):
        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=250, return_tensors="pt").to(device) 
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        input_ids = text_input.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, model.mlm_probability)
        input_ids, labels, masked_indices = model.answer_mask(input_ids, input_ids, model.text_encoder.config.vocab_size,\
                                    image.device, 0, targets=labels,
                                    probability_matrix = probability_matrix)
        mlm_output = model.text_encoder(input_ids, 
                                    attention_mask = text_input.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                    labels = labels)
        answer_mlm_logits = F.softmax(mlm_output.logits[masked_indices].view(len(image),6,30522), \
                                    dim=2).detach().cpu()

        assert len(answer_mlm_logits) == len(image)
        for right_answer_index, answer_logits, id in zip(right_answer_list, answer_mlm_logits, q_ids):
            answer_logits = answer_logits.view(6,30522)
            first_answer_token_log_probs = answer_logits[0][first_index].log().view(-1, 1)

            second_answer_token_log_probs = answer_logits[1][second_index].log().view(-1, 1)
            second_answer_token_log_probs[second_index == 0] = 0
            
            third_answer_token_log_probs = answer_logits[2][third_index].log().view(-1, 1)
            third_answer_token_log_probs[third_index == 0] = 0
            
            fourth_answer_token_log_probs = answer_logits[3][fourth_index].log().view(-1, 1)
            fourth_answer_token_log_probs[fourth_index == 0] = 0

            fifth_answer_token_log_probs = answer_logits[4][fifth_index].log().view(-1, 1)
            fifth_answer_token_log_probs[fifth_index == 0] = 0

            # sixth_answer_token_log_probs = answer_logits[5][sixth_index].log().view(-1, 1)
            # sixth_answer_token_log_probs[sixth_index == 0] = 0

            answer_log_probs = torch.cat([first_answer_token_log_probs,\
                                          second_answer_token_log_probs,\
                                          third_answer_token_log_probs,\
                                          fourth_answer_token_log_probs,\
                                          fifth_answer_token_log_probs,], dim=1).sum(1)
            answer_log_probs = answer_log_probs * gqa_answer_length_list
            pred = answer_log_probs.argmax().item()
            pred = all_answer_dict[str(pred)]
            if pred == right_answer_index:
                results.append([pred, right_answer_index, id.item()])
            test_results.append([pred, right_answer_index,  id.item()])
    print('---------GQA test correct rate---------')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print('total num: 26534.0 test result for Q2A in VCR: 71.3%')
    print('+                                                     +')
    print('                GQA VAL ACC{}%'.format(round(len(results)*100 / len(test_results), 2)) )
    print('+                                                     +')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
