"""Opt-in experiment contexts with the existing optimized decoder and graphs."""
import contextlib

from .cuda_experiments import encoder_provider
from .decoder_kernels import decoder_transform,control_transform
from .graph_memory import decoder_graph_pool
from .graphs import enable_pipeline_graphs,disable_pipeline_graphs


@contextlib.contextmanager
def experimental_pipeline(model,candidate='ptx_rowfusion',out=None):
    with decoder_graph_pool(model),encoder_provider(model,candidate,out) as provider, \
         decoder_transform(model,joint=True,lstm=True,precompute=True,block=1), \
         control_transform(model,storage=True):
        enable_pipeline_graphs(model,max_graphs=4)
        try:yield provider
        finally:disable_pipeline_graphs(model)
