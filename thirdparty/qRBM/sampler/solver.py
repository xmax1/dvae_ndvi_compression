from dwave_sapi2.local import local_connection
from dwave_sapi2.remote import RemoteConnection
from dwave import url, token

remote_connection = RemoteConnection(url, token)
# remote_connection = RemoteConnection(url, token, proxy_url)

solver_names = remote_connection.solver_names()

print 'End'