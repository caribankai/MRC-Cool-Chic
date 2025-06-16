import torch

def get_shape(depth: int, which_latent: int) -> torch.Tensor:
    
    # Depth 0 # # # # # # # # # # #
    if depth==0:
        return torch.tensor(
            [
                            13, 14,
                    20, 21, 22, 23, 24,
                28, 29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
        )
    
    # Depth 1 # # # # # # # # # # #
    elif depth==1:
        if which_latent==0:
            return torch.tensor(
            [
                            13, 
                        21, 22, 23,     
                    29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
            )
    
        if which_latent==1:
            return torch.tensor(
                [           
                        39, 40, 41, 
                            49,                                                                                             
                ]   
            )
        
    # Depth 2 # # # # # # # # # # # #
    elif depth==2:
        if which_latent==0:
            return torch.tensor(
            [
                            13, 
                        21, 22, 23,     
                    29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
            ) 
        if which_latent==1:
            return torch.tensor(
                [           
                            40, 41, 
                            49,                                                                                             
                ]      
            )       
        if which_latent==2:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]    
            )
        
    # Depth 3 # # # # # # # # # # # #
    elif depth==3:
        if which_latent==0:
            return torch.tensor(
            [
                            13, 
                        21, 22, 23,     
                    29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
            ) 
        if which_latent==1:
            return torch.tensor(
                [           
                            40, 41,                                                                                             
                ]   
            )       
        if which_latent==2:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )      
        if which_latent==3:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]      
            )
        
    # Depth 4 # # # # # # # # # # # #
    elif depth==4:
        if which_latent==0:
            return torch.tensor(
            [
                            13, 
                        21, 22, 23,     
                    29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
            )
        if which_latent==1:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]    
            )   
        if which_latent==2:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )   
        if which_latent==3:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )   
        if which_latent==4:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]    
            )
        
    # Depth 5 # # # # # # # # # # #
    elif depth==5:
        if which_latent==0:
            return torch.tensor(
            [
                            13, 
                        21, 22, 23,     
                    29, 30, 31, 32, 33,
                    38, 39, #
            ]
            )
        if which_latent==1:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]   
            )   
        if which_latent==2:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]      
            )   
        if which_latent==3:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )   
        if which_latent==4:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )      
        if which_latent==5:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )

    # Depth 6 # # # # # # # # # # #
    elif depth==6:
        if which_latent==0:
            return torch.tensor(
            [
                        21, 22, 23,     
                    29, 30, 31, 32, 33,
                    38, 39, #
            ]
            )
        if which_latent==1:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]      
            )   
        if which_latent==2:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]   
            )   
        if which_latent==3:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )   
        if which_latent==4:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]     
            )      
        if which_latent==5:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]    
            )
        if which_latent==6:
            return torch.tensor(
                [           
                            40,                                                                                              
                ]   
            )