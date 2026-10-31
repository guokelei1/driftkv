"""Shared joint final-score fitting through the frozen complete query."""
import json
import torch
from experiments.archive.compact_read.calibrate_lightweight_decoder import batches
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.evaluation.cost import CostModel, eager_read


def stack_raw(rows,uids):
    chosen=[rows[uid] for uid in uids]
    cache=HSTUKVCache(torch.cat([x["cache"].k for x in chosen],dim=1),
        torch.cat([x["cache"].v for x in chosen],dim=1),chosen[0]["cache"].seq_len)
    return (cache,torch.cat([x["item_features"] for x in chosen]),
        torch.cat([x["candidates"] for x in chosen]),
        torch.cat([x["query_delta"] for x in chosen]),
        torch.cat([x["teacher_logits"] for x in chosen]))


def fit_joint(model,adapter,refiner,rows,train,validation,*,epochs,batch_size,checks,costs):
    cm=CostModel(model.cfg.hidden_size,len(model.blocks),model.cfg.num_heads,"torch")
    parameters=tuple(refiner.parameters())
    size=sum(p.numel() for p in parameters)
    optimizer=torch.optim.Adam(parameters,lr=.001)
    generator=torch.Generator(device="cpu").manual_seed(17)
    history=[]
    first_gradients=None

    def forward(uids):
        cache,item,ids,delta,wanted=stack_raw(rows,uids)
        # No cached state: codes must reflect this optimizer step and retain
        # the encoder graph, including gradients from downstream query layers.
        state=refiner.encode_cache(adapter,cache,item)
        actual=score(model,cache,ids,delta,
            history_read_fn=refiner.make_history_read(adapter,state))[0]
        extra=adapter.estimate_flops(batch=len(uids),tokens=cache.seq_len,candidates=ids.shape[1])
        query=eager_read(cm,cache.seq_len,queries=ids.shape[1],batch=len(uids))
        return actual,wanted,query+extra["extra_history_read"],extra["token_transform"]

    @torch.no_grad()
    def validate(epoch):
        squared,count=0.,0
        for uids in batches(validation,rows,batch_size):
            actual,wanted,query,encode=forward(uids)
            squared+=float((actual-wanted).square().sum())
            count+=actual.numel()
            costs["joint_validation_query_flops"]+=query
            costs["joint_validation_encoder_flops"]+=encode
            costs["joint_validation_logit_loss_flops"]+=3*actual.numel()
            costs["joint_validation_coordinates_flops"]+=2*size
        mse=squared/count
        history.append(dict(epoch=epoch,validation_teacher_logit_mse=mse,
            normalized_joint_delta_rms=float(refiner.coordinate_rms())))
        costs["joint_coordinate_diagnostics_flops"]+=2*size+4
        return mse

    best,best_epoch=validate(0),0
    best_state={key:value.detach().clone() for key,value in refiner.state_dict().items()}
    for epoch in range(1,epochs+1):
        squared,count=0.,0
        for uids in batches(train,rows,batch_size,generator=generator):
            actual,wanted,query,encode=forward(uids)
            logit_loss=(actual-wanted).square().mean()
            loss=logit_loss+.001*refiner.identity_penalty()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite joint logit loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_gradients is None:
                first_gradients={name:float(p.grad.norm()) for name,p in refiner.named_parameters()}
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite joint gradient")
            optimizer.step()
            squared+=float(logit_loss.detach())*actual.numel()
            count+=actual.numel()
            costs["joint_training_query_forward_flops"]+=query
            costs["joint_training_query_backward_flops_estimate"]+=2*query
            costs["joint_training_encoder_forward_flops"]+=encode
            costs["joint_training_encoder_backward_flops_estimate"]+=2*encode
            costs["joint_training_logit_loss_flops"]+=6*actual.numel()
            costs["joint_coordinates_forward_backward_flops"]+=3*size
            costs["joint_identity_prior_flops"]+=6*size+4
            costs["joint_optimizer_clip_flops_estimate"]+=18*size+10
        if epoch in checks:
            mse=validate(epoch)
            history[-1]["online_training_teacher_logit_mse"]=squared/count
            if mse<best:
                best,best_epoch=mse,epoch
                best_state={key:value.detach().clone() for key,value in refiner.state_dict().items()}
            print(json.dumps(dict(status="joint_validation",epoch=epoch,
                validation_teacher_logit_mse=mse,selected_epoch=best_epoch)),flush=True)
    refiner.load_state_dict(best_state)
    costs["joint_coordinate_diagnostics_flops"]+=2*size+4
    return dict(selected_epoch=best_epoch,validation_history=history,
        first_step_gradient_norms=first_gradients,trainable_parameters=size,
        encoder_parameters=refiner.weight_delta.numel()+refiner.bias_delta.numel(),
        decoder_parameters=refiner.decoder.delta.numel(),
        normalized_joint_delta_rms=float(refiner.coordinate_rms().detach()),
        codes_recomputed_with_grad_every_training_forward=True)
