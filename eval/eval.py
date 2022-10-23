import torch
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
def grounding_test():
    pass

@torch.no_grad()
def vrd_test():
    pass

@torch.no_grad()
def vqa_test():
    pass
