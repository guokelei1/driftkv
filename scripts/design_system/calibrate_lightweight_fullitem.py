"""Full-item arithmetic and checkpoint saving shared by current local-KV fitting."""
import torch
from design_system.calibrate_lightweight import local_flops

SOURCE_PATHS=("scripts/design_system/calibrate_lightweight_fullitem.py",
    "scripts/design_system/calibrate_lightweight.py","scripts/design_system/calibrate_lightweight_decoder.py",
    "scripts/design_two/capture.py","src/hstu_kvcache/design_one/lightweight_fullitem.py",
    "src/hstu_kvcache/design_one/lightweight.py","src/hstu_kvcache/design_two/decoder_refinement.py",
    "src/hstu_kvcache/adaptation/reader.py")


def fullitem_flops(*,batch,length,heads,dim,rank,queries,backward):
    result=local_flops(batch=batch,length=length,heads=heads,dim=dim,
        rank=rank,queries=queries,backward=backward)
    additional_inputs=(heads-1)*dim
    added_mm=2*batch*heads*length*additional_inputs*rank
    result["local_forward_flops"]+=added_mm
    if backward:
        result["local_backward_flops"]+=added_mm
        result["optimizer_clip_flops"]+=18*heads*additional_inputs*rank
    return result


def save_adapter(adapter,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind=adapter.kind,config=adapter.get_config(),
        state_dict={key:value.detach().cpu() for key,value in adapter.state_dict().items()}),path)
