from tqdm import tqdm

import torch
import torch.nn as nn


@torch.no_grad()
def ppl_eval(model, tokenizer, testenc, args):
    print('Evaluating ppl...')
    model.eval()
    model.enable_auto_regressive()
    eval_seq_length = 5000   # fix model max length
    
    eval_iter = 1

    dev = model.lm_head.weight.device
    
    batch_size = args.batch_size
    
    bos = tokenizer.bos_token_id
    
    testenc = testenc.to(dev)
    nlls = []
    for i in range(eval_iter):
        batch_inputs = torch.zeros((batch_size, eval_seq_length), dtype=torch.long, device=dev)
        for j in range(batch_size):
            sample = testenc[:, ((i + j) * eval_seq_length): ((i + j + 1) * eval_seq_length)]
            sample[:, 0] =  bos # ensure the 1st token is <bos>
            batch_inputs[j] = sample
        # input(f"batch.shape: {batch.shape}")
        with torch.no_grad():
            lm_logits = model(batch_inputs).logits
        # print(lm_logits.shape)
        shift_logits = lm_logits[:, :-1, :]
        # print(shift_logits.shape)
        shift_labels = []
        for j in range(batch_size):
            shift_labels.append(batch_inputs[j][1:])
        shift_labels = torch.stack(shift_labels)
        # print(shift_labels.shape)

        # ignore next token prediction after <bos>
        shift_logits = shift_logits[:, 1:, :].contiguous()
        shift_labels = shift_labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * eval_seq_length
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (eval_iter * (eval_seq_length - 1)))
    print(ppl.item())
    return ppl.item()
