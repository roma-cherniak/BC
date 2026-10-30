from utils import load_pairs, load_jepa, load_zond
from checks.check_code import check_code
from checks.check_imagination import check_imagination

test = load_pairs("test")                 
encoder, predictor = load_jepa()
zond = load_zond("zond.pt")



if not check_code(encoder, predictor, zond, test)["code_ok"]:   
    exit("code brocken")

results = {}                               
results.update(check_imagination(encoder, predictor, test))

print(results)
