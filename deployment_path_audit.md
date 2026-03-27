# compliance-checker 路径关联审计（用于服务器部署）

本文档列出 compliance-checker 目录中与本机路径或本机运行环境相关的内容，帮助你在服务器正式运行前快速排查。

## 结论摘要

1. 业务代码中没有写死 Windows 绝对路径（如 D:\\...）。
2. 主要风险来自三类：
- 本地文档中的 file:///d:/... 绝对链接
- .venv 目录内脚本写死了本机虚拟环境路径
- 默认配置使用 localhost 与相对路径，需要在服务器按环境覆盖

## A. 明确与本机绝对路径绑定（需要处理或忽略）

### 1) 规划/说明文档中的 file:///d:/... 绝对路径

这类路径是文档链接，不参与运行，但在服务器上会失效。

- [implementation_plan.md.resolved](implementation_plan.md.resolved#L64)
- [implementation_plan.md.resolved](implementation_plan.md.resolved#L69)
- [implementation_plan.md.resolved](implementation_plan.md.resolved#L87)
- [implementation_plan.md.resolved](implementation_plan.md.resolved#L158)
- [walkthrough.md.resolved](walkthrough.md.resolved#L33)
- [walkthrough.md.resolved](walkthrough.md.resolved#L34)
- [walkthrough.md.resolved](walkthrough.md.resolved#L38)

处理建议：
- 服务器部署可保留这些文档，不影响运行。
- 如果你希望文档在服务器端也可点击，改成仓库相对链接。

### 2) .venv 内激活脚本写死本机路径

这些文件是本地虚拟环境自动生成，不应随代码部署到服务器。

- [.venv/Scripts/activate](.venv/Scripts/activate#L81)
- [.venv/Scripts/activate.bat](.venv/Scripts/activate.bat#L29)
- [.venv/Scripts/activate.csh](.venv/Scripts/activate.csh#L34)
- [.venv/Scripts/activate.fish](.venv/Scripts/activate.fish#L82)
- [.venv/Scripts/activate.nu](.venv/Scripts/activate.nu#L71)

处理建议：
- 不要把 .venv 上传到服务器。
- 服务器上重新创建虚拟环境并安装依赖。

## B. 与本地运行环境相关（需按服务器环境覆盖）

### 1) 默认输出目录是相对路径

- [text/config/settings.py](text/config/settings.py#L55)

当前默认值为 ./compliance_output。它不绑定你的 D 盘，但会依赖进程工作目录。

处理建议：
- 服务器使用环境变量 COMPLIANCE_WORK_DIR 指定绝对目录，例如 /srv/compliance_output。

### 2) 默认 OPA 地址是 localhost

- [text/config/settings.py](text/config/settings.py#L159)

localhost 在服务器上指向服务器自身，若 OPA 不在同机，会连不上。

处理建议：
- 按实际拓扑设置 COMPLIANCE_OPA_URL，例如 http://opa:8181 或内网地址。

### 3) Docker Compose 的本地卷挂载与本地健康检查地址

- [text/docker-compose.yml](text/docker-compose.yml#L20)
- [text/docker-compose.yml](text/docker-compose.yml#L39)
- [text/docker-compose.yml](text/docker-compose.yml#L45)

说明：
- ./compliance_output:/app/compliance_output、./data:/app/data 依赖宿主机当前目录结构。
- localhost 健康检查发生在容器内部，一般可用，但要确保镜像内有 curl。

处理建议：
- 在服务器确认部署目录下存在 data 与 compliance_output。
- 若基础镜像缺少 curl，改为更稳妥的 healthcheck 方式或安装 curl。

## C. 仅测试相关，不影响生产

- [text/tests/test_pipeline.py](text/tests/test_pipeline.py#L15)
- [text/tests/test_pipeline.py](text/tests/test_pipeline.py#L119)

说明：
- 这些是测试注释和不存在路径示例，不进入生产运行链路。

## D. 其他“路径相关但可移植”的代码点

- [text/config/settings.py](text/config/settings.py#L107)
- [text/config/settings.py](text/config/settings.py#L114)

说明：
- 通过 __file__ 推导 keywords.txt 和 patterns.yaml 的路径。
- 这是相对源码位置的可移植写法，通常无需修改。

## E. 上服务器前最小检查清单

1. 排除 .venv、__pycache__、本地临时输出目录。
2. 在服务器设置以下环境变量：
- COMPLIANCE_WORK_DIR
- COMPLIANCE_OPA_URL
- COMPLIANCE_OPA_ENABLED
- COMPLIANCE_QWEN_GUARD_ENABLED
3. 若走 Docker，确认 text 目录下 data 和 compliance_output 挂载目录存在。
4. 如果需要保留 resolved 文档可读性，把其中 file:///d:/... 改成相对路径链接。

## F. 可直接复用的判断

从运行代码角度看，真正需要你在服务器调整的主要是配置项（工作目录、OPA 地址、模型开关），不是业务代码中的硬编码本机路径。
