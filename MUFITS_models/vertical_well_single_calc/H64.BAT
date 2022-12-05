@echo off

set NPROC= 1
set ARGS=

:: path to MUFITS simulator (H64.EXE)
set HDS="C:\Users\kanal\Documents\MUFITS\BIN\H64.EXE"

:: path to MPI
set MPI="C:\Program Files\MPICH2\bin\mpiexec.exe"


call:clear_files

call:runall

goto:eof

======================================================

:runtest
@echo on
%MPI% -n %NPROC% %HDS% %1.RUN %ARGS% > %1.LOG
@echo off
goto:eof

:runall
for /r . %%f in (*.RUN) do ( 
    call:runtest %%~nf
)
goto:eof

:clear_files
del *.SUM /S /Q 
del *.vtu /S /Q
del *.pvd /S /Q
del *.MVS /S /Q
del *.CSV /S /Q
goto:eof