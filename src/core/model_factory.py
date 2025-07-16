from src.core.models import Network, HENetwork, NoEncFusionNetwork, FusionNetwork

class ModelFactory:
    def __init__(self, args):
        self.args = args
    
    def get_single_network(self):
        if self.args.no_enc:
            net = Network(
                dims=self.args.dims,
                args=self.args
            )
        else:
            net = HENetwork(
                dims=self.args.dims,
                args=self.args
            )
        
        return net
    
    def get_fusion_network(self):
        if self.args.no_enc:
            net = NoEncFusionNetwork(
                args=self.args,
                dims_one=self.args.dims[0],
                dims_two=self.args.dims[1]
            )
        else:
            net = FusionNetwork(
                args=self.args,
                dims_one=self.args.dims[0],
                dims_two=self.args.dims[1]
            )
        
        return net 