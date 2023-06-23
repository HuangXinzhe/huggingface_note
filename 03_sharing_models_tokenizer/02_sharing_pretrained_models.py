"""
创建新模型存储库的方法有以下三种：
    1、使用push_to_hub API接口
    2、使用huggingface_hub Python库
    3、使用web界面
"""

"""
1、使用push_to_hub API接口
在notebook中
from huggingface_hub import notebook_login

notebook_login()

在终端中
huggingface-cli login
"""
# from transformers import AutoModelForMaskedLM, AutoTokenizer
#
# checkpoint = "camembert-base"
#
# model = AutoModelForMaskedLM.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# # 创建新存储库名为dummy-model，若存在则会报错，organization为机构名称，use_auth_token为token
# # model.push_to_hub("dummy-model", organization="huggingface", use_auth_token="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")
# model.push_to_hub("dummy-model", use_auth_token="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")

"""
2、使用huggingface_hub Python库
huggingface-cli login

from huggingface_hub import (
    # User management
    login,
    logout,
    whoami,

    # Repository creation and management
    create_repo,
    delete_repo,
    update_repo_visibility,

    # And some methods to retrieve/change information about the content
    list_models,
    list_datasets,
    list_metrics,
    list_repo_files,
    upload_file,
    delete_file,
)
"""
# 创建新库
from huggingface_hub import create_repo

# private 以指定存储库是否应对其他人可见。
# token 如果您想用给定的令牌覆盖存储在缓存中的令牌。
# repo_type 如果你想创建一个数据集或一个空间来替代一个模型。接受的值是dataset和space。
create_repo("dummy-model", organization="huggingface")


"""
3、使用web界面
在网页中创建新库
"""


"""
上传文件
"""
# 方式一
from huggingface_hub import upload_file

# token 如果您想用给定的令牌覆盖存储在缓存中的令牌。
# repo_type 如果你想创建一个数据集或一个空间来替代一个模型。接受的值是dataset和space。
upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)

# 方式二
from huggingface_hub import Repository

# 为了开始使用我们刚刚创建的存储库，我们可以通过克隆远程存储库将其初始化到本地文件夹开始
repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")

# repo.git_pull()  # 拉取最新的代码

# 保存相关的模型
# model.save_pretrained("<path_to_dummy_folder>")
# tokenizer.save_pretrained("<path_to_dummy_folder>")

# 将保存的代码添加提交
# repo.git_add()
# repo.git_commit()
# repo.git_push()

# repo.git_tag()

# 方式三
# 使用git