from huggingface_hub import hf_hub_download

# REPO_ID = "TheBloke/Llama-2-13B-GGML"
# FILENAME = "llama-2-13b.ggmlv3.q4_0.bin"
# LOCAL_DIR = "./models"
# hf_hub_download(
#     repo_id=REPO_ID,
#     filename=FILENAME,
#     local_dir=LOCAL_DIR,
#     local_dir_use_symlinks=False,
# )


# REPO_ID = "TheBloke/Llama-2-13B-GGML"
# FILENAME = "llama-2-13b.ggmlv3.q8_0.bin"
# LOCAL_DIR = "./models"
# hf_hub_download(
#     repo_id=REPO_ID,
#     filename=FILENAME,
#     local_dir=LOCAL_DIR,
#     local_dir_use_symlinks=False,
# )

# REPO_ID = "TheBloke/Llama-2-70B-Chat-GGML"
# FILENAME = "llama-2-70b-chat.ggmlv3.q5_K_S.bin"
# LOCAL_DIR = "./models"
# hf_hub_download(
#     repo_id=REPO_ID,
#     filename=FILENAME,
#     local_dir=LOCAL_DIR,
#     local_dir_use_symlinks=False,
# )

REPO_ID = "TheBloke/WizardLM-13B-V1.2-GGML"
FILENAME = "wizardlm-13b-v1.2.ggmlv3.q4_1.bin"
LOCAL_DIR = "./models"
hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
)
