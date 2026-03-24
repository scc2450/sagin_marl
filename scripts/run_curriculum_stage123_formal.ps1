$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. .\.venv\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"

$updates = 1500
$numEnvs = 12
$vecBackend = "subproc"
$torchThreads = 2
$evalEpisodes = 20

$runRoot = "runs/phase1_actions/curriculum_formal_u1500_subproc12_t2"
$stage1Dir = Join-Path $runRoot "stage1_accel"
$stage2Dir = Join-Path $runRoot "stage2_bw"
$stage3Dir = Join-Path $runRoot "stage3_sat"

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

function Write-LogLine {
    param(
        [string]$RunDir,
        [string]$Message
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    if ($RunDir) {
        $line | Tee-Object -FilePath (Join-Path $RunDir "console.log") -Append
    } else {
        Write-Output $line
    }
}

function Invoke-TrainingStage {
    param(
        [string]$Config,
        [string]$RunDir,
        [string]$InitActor = "",
        [string]$InitCritic = ""
    )

    $pyArgs = @(
        "scripts/train.py",
        "--config", $Config,
        "--updates", "$updates",
        "--run_dir", $RunDir,
        "--num_envs", "$numEnvs",
        "--vec_backend", $vecBackend,
        "--torch_threads", "$torchThreads"
    )
    if ($InitActor) {
        $pyArgs += @("--init_actor", $InitActor)
    }
    if ($InitCritic) {
        $pyArgs += @("--init_critic", $InitCritic)
    }

    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    Write-LogLine -RunDir $RunDir -Message "TRAIN $Config -> $RunDir"
    python -u @pyArgs 2>&1 | Tee-Object -FilePath (Join-Path $RunDir "console.log") -Append
}

function Invoke-FinalEval {
    param(
        [string]$Config,
        [string]$RunDir,
        [string]$Checkpoint,
        [int]$EpisodeSeedBase
    )

    $trainedOut = Join-Path $RunDir "eval_trained_final.csv"
    $baselineOut = Join-Path $RunDir "eval_queue_aware_final.csv"

    Write-LogLine -RunDir $RunDir -Message "EVAL TRAINED $Config -> $trainedOut"
    python -u scripts/evaluate.py `
        --config $Config `
        --checkpoint $Checkpoint `
        --episodes $evalEpisodes `
        --episode_seed_base $EpisodeSeedBase `
        --out $trainedOut 2>&1 | Tee-Object -FilePath (Join-Path $RunDir "console.log") -Append

    Write-LogLine -RunDir $RunDir -Message "EVAL BASELINE queue_aware $Config -> $baselineOut"
    python -u scripts/evaluate.py `
        --config $Config `
        --baseline queue_aware `
        --episodes $evalEpisodes `
        --episode_seed_base $EpisodeSeedBase `
        --out $baselineOut 2>&1 | Tee-Object -FilePath (Join-Path $RunDir "console.log") -Append
}

Invoke-TrainingStage `
    -Config "configs/phase1_actions_curriculum_stage1_accel.yaml" `
    -RunDir $stage1Dir
Invoke-FinalEval `
    -Config "configs/phase1_actions_curriculum_stage1_accel.yaml" `
    -RunDir $stage1Dir `
    -Checkpoint (Join-Path $stage1Dir "actor.pt") `
    -EpisodeSeedBase 62000

Invoke-TrainingStage `
    -Config "configs/phase1_actions_curriculum_stage2_bw.yaml" `
    -RunDir $stage2Dir `
    -InitActor (Join-Path $stage1Dir "actor.pt") `
    -InitCritic (Join-Path $stage1Dir "critic.pt")
Invoke-FinalEval `
    -Config "configs/phase1_actions_curriculum_stage2_bw.yaml" `
    -RunDir $stage2Dir `
    -Checkpoint (Join-Path $stage2Dir "actor.pt") `
    -EpisodeSeedBase 72000

Invoke-TrainingStage `
    -Config "configs/phase1_actions_curriculum_stage3_sat.yaml" `
    -RunDir $stage3Dir `
    -InitActor (Join-Path $stage2Dir "actor.pt") `
    -InitCritic (Join-Path $stage2Dir "critic.pt")
Invoke-FinalEval `
    -Config "configs/phase1_actions_curriculum_stage3_sat.yaml" `
    -RunDir $stage3Dir `
    -Checkpoint (Join-Path $stage3Dir "actor.pt") `
    -EpisodeSeedBase 82000

Write-LogLine -RunDir $runRoot -Message "ALL STAGES COMPLETED"
