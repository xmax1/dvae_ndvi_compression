from dwave_sapi2.remote import RemoteConnection
from dwave import url, token
from dwave_sapi2.util import get_hardware_adjacency


solver = 'C16'
remote_connection = RemoteConnection(url, token)
# solver_names = remote_connection.solver_names()
solver = remote_connection.get_solver(solver)
adj = get_hardware_adjacency(solver)


