import onedrivesdk

redirect_uri = 'http://localhost:8080/'
client_secret = 'blv8Q~KaGO.sfhNakbKqA1IUKgwQTecaiDAe0bFB'
client_id='e23db896-13dd-49d9-9b85-734f8af501fe'
api_base_url='https://api.onedrive.com/v1.0/'
scopes=['wl.signin', 'wl.offline_access', 'onedrive.readwrite']

http_provider = onedrivesdk.HttpProvider()
auth_provider = onedrivesdk.AuthProvider(
    http_provider=http_provider,
    client_id=client_id,
    scopes=scopes)

client = onedrivesdk.OneDriveClient(api_base_url, auth_provider, http_provider)
auth_url = client.auth_provider.get_auth_url(redirect_uri)
# Ask for the code
print('Paste this URL into your browser, approve the app\'s access.')
print('Copy everything in the address bar after "code=", and paste it below.')
print(auth_url)
code = raw_input('Paste code here: ')

client.auth_provider.authenticate(code, redirect_uri, client_secret)