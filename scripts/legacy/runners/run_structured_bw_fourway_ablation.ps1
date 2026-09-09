param(
    [Parameter(Mandatory = $true)]
    [string]$InitActor,
    [Parameter(Mandatory = $true)]
    [string]$InitCritic,
    [string]$PythonExe = ".\\.venv\\Scripts\\python.exe",
    [string]$RunRoot = "runs\\structured_bw_fourway_20260404",
    [int]$Updates = 100,
    [int]$NumEnvs = 8,
    [string]$VecBackend = "sync",
    [string]$Device = "auto",
    [int]$HiddenDim = 128,
    [int]$EmbedDim = 64
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$jobs = @(
    @{
        Name = "bwonly_learnedpartners"
        Config = "configs\\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995_bwonly_learnedpartners.yaml"
    },
    @{
        Name = "bwonly_learnedpartners_user0"
        Config = "configs\\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995_bwonly_learnedpartners_user0.yaml"
    },
    @{
        Name = "bwonly_clusterpartners"
        Config = "configs\\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995_bwonly_clusterpartners.yaml"
    },
    @{
        Name = "bwonly_clusterpartners_user0"
        Config = "configs\\phase1_actions_curriculum_joint_3heads_fading_interference_ka_vsat_joint_puremappo_criticdecoupled_diag_timeline_structured_step250_gamma0995_bwonly_clusterpartners_user0.yaml"
    }
)

foreach ($job in $jobs) {
    $runDir = Join-Path $RunRoot $job.Name
    Write-Host "=== $($job.Name) ===" -ForegroundColor Cyan
    & $PythonExe scripts/train_structured.py `
        --config $job.Config `
        --run_dir $runDir `
        --updates $Updates `
        --num_envs $NumEnvs `
        --vec_backend $VecBackend `
        --device $Device `
        --hidden_dim $HiddenDim `
        --embed_dim $EmbedDim `
        --init_actor $InitActor `
        --init_critic $InitCritic
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $($job.Name) with exit code $LASTEXITCODE"
    }
}
