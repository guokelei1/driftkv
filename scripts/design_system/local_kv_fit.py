"""Cached layer-local Key/Value fitting; the backbone and teacher run once."""
import json
import math

import torch
from torch.nn import functional as F

from design_system.calibrate_lightweight import feature_statistics
from design_system.calibrate_lightweight_fullitem import fullitem_flops
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.design_one.lightweight import LightweightValueState
from read_correction_2026_09.cost import CostModel, eager_read, teacher_history_read


@torch.no_grad()
def capture_layer(model, adapter, rows, uids, index, device, costs):
    records=[]
    h,d,r=adapter.heads,adapter.head_dim,adapter.rank
    cm=CostModel(model.cfg.hidden_size,len(model.blocks),h,"torch")
    for uid in uids:
        row=rows[uid]
        cache,teacher=row["parent"].to(device),row["teacher"].to(device)
        item,n=row["item_features"].to(device),cache.seq_len
        codes=torch.zeros((adapter.num_layers,1,n,h,r),device=device)
        for lower in range(index):
            codes[lower]=adapter.layers[lower].encode(cache.k[lower],cache.v[lower],item)
        lower_read=adapter.make_history_read(LightweightValueState(codes,n),fitted_layers=index)
        captured={}

        def read(layer, attention, q, k, v):
            if layer!=index:return lower_read(layer,attention,q,k,v)
            keys=k.reshape(1,n,h,d).transpose(1,2)
            values=v.reshape(1,n,h,d).transpose(1,2)
            raw=q@keys.transpose(-2,-1)*attention.scale
            native=attention._activate(raw)@values
            wanted=history_read(attention,q,teacher.k[index],teacher.v[index])
            captured.update(q=q.squeeze(0),raw_scores=raw.squeeze(0),
                native_rate=(native/n).squeeze(0),teacher_rate=(wanted/n).squeeze(0),
                attention_scale=float(attention.scale))
            return native

        ids=row["candidates"].to(device)[None]
        score(model,cache,ids,torch.tensor([row["query_delta"]],device=device),history_read_fn=read)
        k=cache.k[index].reshape(n,h,d)
        v=cache.v[index].reshape(n,h,d)
        features=torch.cat((k,v,item[0,:,None,:].expand(n,h,h*d)),dim=-1)
        records.append(dict(length=n,features=features,v=v.transpose(0,1),**captured))
        q=ids.shape[1]
        extra=adapter.estimate_flops(tokens=n,candidates=q)
        costs["query_capture_native_flops"]+=eager_read(cm,n,queries=q)
        costs["same_query_teacher_read_flops"]+=teacher_history_read(cm,n,queries=q)
        costs["query_capture_lower_layer_encoding_flops"]+=extra["token_transform"]*index//adapter.num_layers
        costs["query_capture_lower_layer_read_flops"]+=extra["extra_history_read"]*index//adapter.num_layers
        costs["capture_rate_normalization_flops"]+=2*h*q*d
    b,n,q=len(records),max(x["length"] for x in records),records[0]["q"].shape[1]
    data=dict(features=torch.zeros((b,n,h,(h+2)*d),device=device),
        v=torch.zeros((b,h,n,d),device=device),raw_scores=torch.zeros((b,h,q,n),device=device),
        q=torch.stack([x["q"] for x in records]),
        native_rate=torch.stack([x["native_rate"] for x in records]),
        teacher_rate=torch.stack([x["teacher_rate"] for x in records]),
        lengths=torch.tensor([x["length"] for x in records],device=device),
        attention_scale=records[0]["attention_scale"])
    for i,record in enumerate(records):
        length=record["length"]
        data["features"][i,:length]=record["features"]
        data["v"][i,:,:length]=record["v"]
        data["raw_scores"][i,:,:,:length]=record["raw_scores"]
    return data


def cached_prediction(layer, features, q, raw_scores, v, lengths,
                      native_rate, output_scale, key_scale, attention_scale):
    z=F.silu(features.transpose(1,2)@layer.input_weight[None]+layer.input_bias[None,:,None])
    dk=layer.key_decoder*key_scale
    scores=raw_scores+(q@dk.transpose(-2,-1)[None])@z.transpose(-2,-1)*attention_scale
    mask=torch.arange(z.shape[2],device=z.device)[None,:]<lengths[:,None]
    weights=(F.elu(scores)+1)*mask[:,None,None,:]/lengths[:,None,None,None]
    return ((weights@v-native_rate)/output_scale[None,:,None,None]
            +(weights@z)@layer.decoder[None])


def local_flops(*,batch,length,heads,dim,rank,queries,backward):
    result=fullitem_flops(batch=batch,length=length,heads=heads,dim=dim,
        rank=rank,queries=queries,backward=backward)
    b,n,h,d,r,q=batch,length,heads,dim,rank,queries
    pairs,outputs,code=b*h*q*n,b*h*q*d,b*h*n*r
    project,latent,native=2*b*h*q*d*r,2*pairs*r,2*pairs*d
    # Scale, add, ELU+1 (3), mask and divide N; native rate subtraction and scale.
    result["local_key_attention_forward_flops"]=project+latent+native+7*pairs+2*outputs
    result["local_key_coordinate_forward_flops"]=h*r*d
    if backward:
        # AV dweights; AZ dweights; score product both gradients; Dk dW.
        result["local_key_attention_backward_flops"]=project+3*latent+native+5*pairs+outputs+code
        result["local_key_coordinate_backward_flops"]=h*r*d
        result["local_key_optimizer_clip_flops"]=18*h*r*d
    return result


def fit_layer(layer,data,ntrain,*,epochs,batch_size,checks,costs):
    features=data["features"]
    b,n,h,width=features.shape
    d,r,q=layer.head_dim,layer.rank,data["q"].shape[2]
    with torch.no_grad():
        center,scale=feature_statistics(features[:ntrain],data["lengths"][:ntrain])
        layer.input_center.copy_(center);layer.input_scale.copy_(scale)
        normalized=(features-center)/scale
        residual=data["teacher_rate"]-data["native_rate"]
        output_scale=residual[:ntrain].square().mean((0,2,3)).sqrt().clamp_min(1e-6)
        wanted=residual/output_scale[None,:,None,None]
        initial=F.silu(normalized[:ntrain].transpose(1,2)@layer.input_weight[None]
                       +layer.input_bias[None,:,None])
        mask=(torch.arange(n,device=features.device)[None,:]<data["lengths"][:ntrain,None])
        tokens=data["lengths"][:ntrain].sum()
        key_rms=((features[:ntrain,:,:,:d].square()*mask[:,:,None,None]).sum((0,1,3))/(tokens*d)).sqrt().clamp_min(1e-6)
        code_rms=((initial.square()*mask[:,None,:,None]).sum((0,2,3))/(tokens*r)).sqrt()
        denominator=torch.where(code_rms>1e-6,code_rms,torch.ones_like(code_rms))
        key_scale=(key_rms/(math.sqrt(r)*denominator))[:,None,None]
        energy=float(data["teacher_rate"][ntrain:].square().mean())
    costs["feature_statistics_flops"]+=4*ntrain*n*h*width+5*h*width
    costs["feature_normalization_flops"]+=2*b*n*h*width
    costs["target_statistics_flops"]+=2*b*h*q*d+2*ntrain*h*q*d+2*h
    costs["validation_teacher_energy_flops"]+=2*(b-ntrain)*h*q*d+1
    costs["initial_fit_encoder_flops"]+=ntrain*h*n*(2*width*r+2*r)
    costs["local_key_statistics_flops"]+=3*ntrain*n*h*(d+r)+8*h+ntrain
    optimizer=torch.optim.Adam(layer.parameters(),lr=.001)
    generator=torch.Generator(device="cpu").manual_seed(17)
    history=[];gradient_samples={}

    def predict(indices):
        return cached_prediction(layer,normalized[indices],data["q"][indices],
            data["raw_scores"][indices],data["v"][indices],data["lengths"][indices],
            data["native_rate"][indices],output_scale,key_scale,data["attention_scale"])

    @torch.no_grad()
    def validate(epoch):
        squared,count=0.,0
        for start in range(ntrain,b,batch_size):
            stop=min(start+batch_size,b)
            difference=(predict(slice(start,stop))-wanted[start:stop])*output_scale[None,:,None,None]
            squared+=float(difference.square().sum());count+=difference.numel()
            for name,value in local_flops(batch=stop-start,length=n,heads=h,dim=d,
                    rank=r,queries=q,backward=False).items():costs["validation_"+name]+=value
            costs["validation_physical_scale_flops"]+=difference.numel()
        mse=squared/count
        history.append(dict(epoch=epoch,validation_read_rate_mse=mse,
            validation_relative_teacher_mse=mse/max(energy,1e-30)))
        return mse

    best,best_epoch=validate(0),0
    best_state={name:value.detach().clone() for name,value in layer.state_dict().items()}
    step=0
    for epoch in range(1,epochs+1):
        order=torch.randperm(ntrain,generator=generator).to(features.device)
        for start in range(0,ntrain,batch_size):
            indices=order[start:start+batch_size]
            loss=(predict(indices)-wanted[indices]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError("nonfinite local KV loss")
            optimizer.zero_grad(set_to_none=True);loss.backward();step+=1
            if step<=2:
                gradient_samples[str(step)]={name:float(p.grad.norm()) for name,p in layer.named_parameters()}
                costs["local_gradient_diagnostic_flops"]+=sum(2*p.numel()+1 for p in layer.parameters())
            norm=torch.nn.utils.clip_grad_norm_(layer.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError("nonfinite local KV gradient")
            optimizer.step()
            for name,value in local_flops(batch=len(indices),length=n,heads=h,dim=d,
                    rank=r,queries=q,backward=True).items():costs[name]+=value
        if epoch in checks:
            mse=validate(epoch)
            if mse<best:
                best,best_epoch=mse,epoch
                best_state={name:value.detach().clone() for name,value in layer.state_dict().items()}
            print(json.dumps(dict(status="local_kv_validation",epoch=epoch,
                validation_mse=mse,selected_epoch=best_epoch)),flush=True)
    layer.load_state_dict(best_state)
    with torch.no_grad():
        layer.decoder.mul_(output_scale[:,None,None]);layer.key_decoder.mul_(key_scale)
    costs["decoder_scale_fold_flops"]+=2*h*r*d
    return dict(selected_epoch=best_epoch,validation_history=history,
        first_step_gradient_norms=gradient_samples["1"],first_two_gradient_norms=gradient_samples,
        output_scale=output_scale.cpu().tolist(),key_scale=key_scale[:,0,0].cpu().tolist(),
        initial_code_rms=code_rms.cpu().tolist(),fit_key_rms=key_rms.cpu().tolist(),
        teacher_rate_energy=energy,train_users=ntrain,validation_users=b-ntrain,
        input_width=width,padded_length=n,actual_history_lengths=data["lengths"].cpu().tolist())
