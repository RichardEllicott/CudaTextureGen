# Windows Building

Windows is a bit more complicated, a lot of enviroment variables have been hardcoded so might rely on my compiler version etc

important info in:

- BUILD.ps1
- toolchain-msvc.cmake



## I had it working, this commit

https://github.com/RichardEllicott/hello_cuda/commit/5e2dbfeded167a2b979607f8402aab4480a17f34

to get that commit back, go to my windo

```powershell
git clone --recurse-submodules https://github.com/RichardEllicott/hello_cuda.git # clone repo, get the submodules, this will go in the hello_cuda folder
cd hello_cuda # nav into dir
git checkout 5e2dbfeded167a2b979607f8402aab4480a17f34 # revert to that working one
rm -Recurse -Force .git # (Optional) If you want to sever Git ties and keep just a snapshot
```



