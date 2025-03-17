import torch

#test youwei 
class PositionEmbedding(torch.nn.Module):
   def __init__(self):
      super().init__()

      # pos est le 
      def get_pe(pos,i,d_model):
         fenmu = 1e4**(i/d_model)
         pe = pos/fenmu



# test num 2p
# test 3