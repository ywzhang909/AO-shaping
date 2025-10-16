1. mount data directory

```bash
sudo mount -t cifs -o user=tifo,password=TIFO1234,uid=tifo,gid=tifo,iocharset=utf8,vers=3.0 //10.10.0.53/storage/AO_data data
```