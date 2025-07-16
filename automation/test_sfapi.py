from sfapi_client import Client
from sfapi_client.compute import Machine
client = Client(key='./priv_key.pem')
perlmutter = client.compute(Machine.perlmutter)
print(perlmutter.ls('/global/cfs/cdirs/m558'))
