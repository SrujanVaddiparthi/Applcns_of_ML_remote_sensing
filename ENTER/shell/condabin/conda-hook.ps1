$Env:CONDA_EXE = "/Users/wangtiles/Applcns_of_ML_remote_sensing/ENTER/bin/conda"
$Env:_CE_M = $null
$Env:_CE_CONDA = $null
$Env:_CONDA_ROOT = "/Users/wangtiles/Applcns_of_ML_remote_sensing/ENTER"
$Env:_CONDA_EXE = "/Users/wangtiles/Applcns_of_ML_remote_sensing/ENTER/bin/conda"
$CondaModuleArgs = @{ChangePs1 = $True}
Import-Module "$Env:_CONDA_ROOT\shell\condabin\Conda.psm1" -ArgumentList $CondaModuleArgs

Remove-Variable CondaModuleArgs