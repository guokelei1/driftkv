"""Pure FLOP accounting for head-local 32 -> W -> 32 residual SiLU maps.

No elapsed seconds, GPU throughput or fitted timing coefficients enter here.
Matrices use 2*m*n*k. Pointwise counts follow the expressions in kv_map.py;
exp, reciprocal and sqrt each count as one, consistent with the native ledger.
"""
CHANNELS = 2*10*10  # independent K/V x layers x heads
DIM = 32

def parameter_count(width, channels=CHANNELS, dim=DIM):
    return channels*(2*dim*width+width+dim)

def deployment_flops(rows,width,channels=CHANNELS,dim=DIM):
    # Input normalization: 2D; two GEMMs: 4DW; biases: W+D;
    # SiLU: 5W; output denormalization: 2D; native residual: D.
    return channels*rows*(4*dim*width+6*width+6*dim)

def step_flops(batch,width,channels=CHANNELS,dim=DIM):
    params=parameter_count(width,channels,dim)
    return dict(
        forward_gemm=channels*batch*4*dim*width,
        forward_pointwise=channels*batch*(6*width+dim),
        backward_gemm=channels*batch*6*dim*width,
        backward_pointwise=channels*(batch*9*width+(batch-1)*(width+dim)),
        loss_forward_backward=channels*batch*5*dim,
        optimizer=14*params+48)

def fitting_flops(observations,width,epochs,batch_rows,teacher_flops,channels=CHANNELS,dim=DIM):
    """All epochs and the real final partial batch, with no input gradients.

    Forward has two GEMMs. Backward computes dW2,dH,dW1 (three GEMMs), not
    dX: frozen backbone K/V are constants. AdamW uses foreach=False/fused=False:
    14 elementwise arithmetic operations/parameter and 12 scalars/tensor for
    four parameter tensors. RNG/indexing/transfers/comparisons are separate.
    """
    full,remainder=divmod(observations,batch_rows)
    batches=[(batch_rows,full)]+([(remainder,1)] if remainder else [])
    parts={key:0 for key in step_flops(1,width,channels,dim)}
    steps=0
    for batch,count in batches:
        if not count:
            continue
        steps+=epochs*count
        for key,value in step_flops(batch,width,channels,dim).items():
            parts[key]+=epochs*count*value
    # Teacher-old residual and explicit two-pass mean/std normalization for
    # X,Y: (1+5+5)*G*M*D plus one sqrt per coordinate/array.
    normalization=11*channels*observations*dim+2*channels*dim
    initialization=channels*dim*width+2  # W1 random draw scaling, sqrt(2/D)
    # Before/after normalized MSE on the same fixed prerelease rows; retained
    # diagnostics are charged, and never select a checkpoint using test data.
    probe_rows=min(2048,observations)
    diagnostics=2*channels*probe_rows*(4*dim*width+6*width+4*dim)
    result=dict(teacher_source_preparation=int(teacher_flops),normalization=normalization,
        initialization=initialization,fit_diagnostics=diagnostics,**parts)
    return dict(parts=result,total_flops=sum(result.values()),observations=observations,
        epochs=epochs,steps=steps,training_rows=epochs*observations,parameters=parameter_count(width,channels,dim),
        arithmetic='Analytical FLOPs, FMA=2; actual GEMM shapes, explicit SiLU/MSE/AdamW expressions; RNG/indexing/copies excluded, not timed FLOPs')
