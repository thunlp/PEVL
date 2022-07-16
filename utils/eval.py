import torch
from tqdm import tqdm

@torch.no_grad()
def vcr_validate(model, data_loader, tokenzier, device, setting):
    model.eval()
    total_count = 0.0
    right_count = 0.0
    for i, (images, text, labels, _) in tqdm(enumerate(data_loader)):
        images = images.to(device, non_blocking=True)
        text = tokenzier(text, padding='longest', truncation=True, max_length=160, return_tensors="pt").to(device) 
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        image_embeds = model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
        text_output = model.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                         return_dict = True, mode = 'text') 
        text_embeds = text_output.last_hidden_state
        output_pos = model.text_encoder(encoder_embeds = text_embeds, 
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
        for x,y in zip(choice, labels):
            if x==y:
                right_count+=1
            test_count+=1
    print(f"test result for {setting} in VCR: {round(right_count/total_count, 3)*100.0}%")

@torch.no_grad()
def grounding_test():
    pass

@torch.no_grad()
def vrd_test():
    pass

@torch.no_grad()
def vqa_test():
    pass

