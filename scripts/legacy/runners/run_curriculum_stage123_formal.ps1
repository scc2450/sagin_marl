$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. .\.venv\Scripts\Activate.ps1
$env:PYTHONUNBUFFERED = "1"

$updatesStage1 = 1500
$updatesStage2 = 1500
$updatesStage3 = 1600
$numEnvs = 12
$vecBackend = "sync"
$torchThreads = 2
$evalEpisodes = 20

$runRoot = "runs/phase1_actions/curriculum_formal_u1500_sync12_t2"
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
        [int]$Updates,
        [string]$InitActor = "",
        [string]$InitCritic = ""
    )

    $pyArgs = @(
        "scripts/train.py",
        "--config", $Config,
        "--updates", "$Updates",
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

function Resolve-CheckpointPath {
    param(
        [string]$RunDir,
        [string]$Stem
    )

    $bestPath = Join-Path $RunDir "${Stem}_best.pt"
    if (Test-Path $bestPath) {
        return $bestPath
    }
    return (Join-Path $RunDir "${Stem}.pt")
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
    -RunDir $stage1Dir `
    -Updates $updatesStage1
Invoke-FinalEval `
    -Config "configs/phase1_actions_curriculum_stage1_accel.yaml" `
    -RunDir $stage1Dir `
    -Checkpoint (Resolve-CheckpointPath -RunDir $stage1Dir -Stem "actor") `
    -EpisodeSeedBase 62000

Invoke-TrainingStage `
    -Config "configs/phase1_actions_curriculum_stage2_bw.yaml" `
    -RunDir $stage2Dir `
    -Updates $updatesStage2 `
    -InitActor (Resolve-CheckpointPath -RunDir $stage1Dir -Stem "actor") `
    -InitCritic (Resolve-CheckpointPath -RunDir $stage1Dir -Stem "critic")
Invoke-FinalEval `
    -Config "configs/phase1_actions_curriculum_stage2_bw.yaml" `
    -RunDir $stage2Dir `
    -Checkpoint (Resolve-CheckpointPath -RunDir $stage2Dir -Stem "actor") `
    -EpisodeSeedBase 72000

Invoke-TrainingStage `
    -Config "configs/phase1_actions_curriculum_stage3_sat.yaml" `
    -RunDir $stage3Dir `
    -Updates $updatesStage3 `
    -InitActor (Resolve-CheckpointPath -RunDir $stage2Dir -Stem "actor") `
    -InitCritic (Resolve-CheckpointPath -RunDir $stage2Dir -Stem "critic")
Invoke-FinalEval `
    -Config "configs/phase1_actions_curriculum_stage3_sat.yaml" `
    -RunDir $stage3Dir `
    -Checkpoint (Resolve-CheckpointPath -RunDir $stage3Dir -Stem "actor") `
    -EpisodeSeedBase 82000

Write-LogLine -RunDir $runRoot -Message "ALL STAGES COMPLETED"
