"""HTTP API layer: request şemaları + merkezi hata→response map'i.

Faz B'nin amacı: app.py'deki route handler'ları parsing/validation
yükünden arındırmak.

    - schemas.py : pydantic request modelleri — validation tek yerde
    - errors.py  : RagArtError → HTTP response map'i tek yerde

Route handler artık "happy path"e odaklanır; geçersiz girdi ve domain
hataları kenara, tek noktaya toplanır.
"""
