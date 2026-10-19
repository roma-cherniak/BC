import torch





def sigreg(z):
    A = torch.randn(z.shape[1], 1024, device = z.device)            
    A = A / A.norm(dim=0)

    x = z @ A 

    t = torch.linspace(0, 3, 17, device = z.device)         
    xt = x.unsqueeze(-1) * t    

    cos_mean = xt.cos().mean(dim=0)      
    sin_mean = xt.sin().mean(dim=0)      
    ideal = torch.exp(-t ** 2 / 2)       
    error = (cos_mean - ideal) ** 2 + sin_mean ** 2 

    weights = torch.full((17,), 2 * 3 / 16 , device = z.device)
    weights[[0, -1]] = 3 / 16
    weights = weights * ideal 

    result = (error @ weights) * len(z)  
    result = result.mean()  
    return result 


if __name__ == "__main__":
    print(sigreg(torch.randn(256, 32)))
    print(sigreg(torch.zeros(256, 32)))
    print(sigreg(torch.randn(256, 32) + 3))
    print(sigreg(torch.randn(256, 32) * 5))
