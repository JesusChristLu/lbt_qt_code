# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/12
# __author:       Lang Zhu
# __corporation:  OriginQuantum
# from markdown.extensions.Extension import Extension

from xml.etree.ElementTree import Element

from markdown import extensions
from markdown.treeprocessors import Treeprocessor


class BootstrapTreeprocessor(Treeprocessor):
    """
    the markdown package extension
    """

    def run(self, node):

        for i in range(len(node)):
            if node[i].tag in ["h1", "h2", "h2", "h3", "h4", "h5", "h6"]:
                # child.text = """<div style="background-color:yellow">{}</div>""".format(child.text)
                node[i].set("class", "md-end-block md-heading")
            elif node[i].tag == "table":
                node[i].set("class", "md-table")
                new_element = Element("figure",
                                      attrib={"class": "md-table-fig"})
                new_element.append(node[i])
                node[i] = new_element
            # elif child.tag == 'img':
            #    child.set("class","img-fluid")
        return node


class BootStrapExtension(extensions.Extension):
    """
    the markdown package extension
    """

    def extendMarkdown(self, md):
        """
        """
        md.registerExtension(self)
        self.processor = BootstrapTreeprocessor()
        self.processor.md = md
        self.processor.config = self.getConfigs()
        # md.treeprocessors.add('bootstrap', self.processor, '_end')
        md.treeprocessors.register(self.processor, 'bootstrap', 25)
