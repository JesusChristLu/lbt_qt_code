# MD Readme

## 概述

Markdown模块用于生成md格式报告,并提供工具和可选主题,可将报告转换为pdf和html格式.
模块生成的实验报告有如下三个特点:

1. 可选报告主题
2. 可定制报告内容详细程度
3. 报告均采用单文件模式,便于保存和分享

项目代码可分为两大模块: `md生成模块`和`Converter转换器`

### md生成模块

![项目代码结构图](markdown%20结构图.svg)

结构由生成器和解析器两大部分组成.

解析器`Parser`用于对接`visage`和`invoker`接口,获取对应的文档原始信息,例如`DAG id`和`Experiment ID`.
然后通过预设的流程获取信息,按照特定格式解析成对应生成器可识别的信息块块后,创建生成器`Generator`来生成md文档.

### Converter转换器

转换器模块封装了`Converter`类用于转换`pdf`和`html`文档,在不进行转换方式迭代的情况下,无需更改内部实现.

Converter模块文件目录如下:

```shell
├─extensions
│  └─democss.py
├─style
│  ├─base
│  └─themes
├─wkhtmltox
│  ├─bin
│  ├─css
│  └─include
│      └─wkhtmltox

```

+ extensions目录存放了用于适配html主题的`markdown`库的自定义插件.
+ style模块用于存放已开发适配的主题文件,目前有`dark` 和`white`
  双色主题.如果需要新增主题,可将新主题配色css文件放至`themes`目录下.
+ `wkhtmltox`文件夹存放了`pdfkit`打包方式所需的依赖文件.

## 使用教程

### md -> html

1. 可直接使用`converter_md_to_html`函数实现
   ```python
   from pyqcat_visage.md.converter import converter_md_to_html, html_add_theme
   md_doc = "# title\ntest"
   # 不添加extensions生成的会导致部分高级md语法无法正确转换,例如table.
   html_doc =converter_md_to_html(md_doc=md_doc)
   
   # 适配table, toc等语法
   extensions = [
            'markdown.extensions.toc', 'markdown.extensions.tables',
            'markdown.extensions.fenced_code'
        ]
   html_doc =converter_md_to_html(md_doc, extensions)
   
   # html 增加主题
   html_doc = html_add_theme(html_doc)
    ```
2. 通过`Converter`类实现
   ```python
   from pyqcat_visage.md.converter import Converter
   md_doc = "# title\ntest"
   converter = Converter()
   converter.option.hold_pdf = False
   # 选择主题
   
   converter.execute(md_doc=md_doc)
   html_doc = converter.doc_html
    ```

推荐采用`Converter`类实现.

### md -> pdf

`markdown`转`pdf`, 首先需要将`markdown`转换为**html**然后将`html_doc`转换为`pdf`.

1. 直接使用html_to_pdf函数
   ```python
   from pyqcat_visage.md.converter import html_to_pdf
   html_doc = "any html"
   pdf_bytes =html_to_pdf(html_doc)
    ```
2. 采用`Converter`类实现
    ```python
   from pyqcat_visage.md.converter import Converter
   html_doc = "any html"

   converter = Converter()
   # 选择主题
   converter.execute(md_doc=md_doc)
   pdf_doc = converter.doc_pdf
    ```

### 生成实验报告

md模块中`magic`文件中封装了visage生成实验报告调用接口`execute`,生成实验报告调用接口,传入实验或DAG的id,并且传入对应的类型,即可生成实验报告.

```python
from pyqcat_visage.md import execute

dag_id = "6360d36178c96f5d55065e22"
# set the report file type.
file_type = "pdf"
# set the report color theme.
theme = "white"
# get pdf report.
dag_report_pdf = execute(dag_id, "dag", theme=theme, save_type=file_type)
# save
with open("test.pdf", "wb") as f:
    f.write(dag_report_pdf)
```
