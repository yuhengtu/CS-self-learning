# URL网址   protocol :// hostname[:port] / path / [;parameters][?query]#fragment
# chromium  谷歌和edge的核心 软件开发难度天花板 开源
# 动态 静态生成
# 呈现在客户端的网页通常是HTML格式的编码，可以通过浏览器提供的“审查元素”功能快速定位到相关位置。
# html教程 https://www.w3school.com.cn/html/index.asp
# requests,scrapy,urllib  爬虫requests https://requests.readthedocs.io/zh_CN/latest/user/quickstart.html
# beautifulsoup   网页解析    selenium自动化措施，模仿人的鼠标（应对淘宝等反爬虫）
# 一般get 和 post  用get为主

import requests
url = 'https://www.baidu.com'
response = requests.get(url) #创建response（响应）对象
response.encoding = 'utf-8'
# r.encoding = r.apparent_encoding
print(response.text) #text对应文本形式

# img_url = 'https://www.cnblogs.com/skins/CodingLife/images/title-yellow.png'
# r = requests.get(img_url)
# with open('img.png', 'wb+') as f:# w写 b指二进制
#     f.write(r.content) #text对应二进制形式

# status_code属性用来获取HTTP访问的状态码，200代表正确，404代表资源不存在，403代表禁止访问，500代表服务器故障等
#  print（r.status_code）

# print(requests.get())
# print(requests.head())
# print(requests.post())

# 头信息
# 通过浏览器提供的开发者工具（如chrome）
# 可以查看到网页的head信息。
# 头信息中最需要了解的是Host、User-Agent、Referer和Cookie
# User-Agent是浏览器标识头，许多的网站根据这个字段来对爬虫做初步的判断，Requests默认不设置这个字段

# 最基本的反爬虫 伪装headers头信息为真人
# import requests
# url = 'https://www.seu.edu.cn/_upload/tpl/04/82/1154/template1154/images/logo.png’
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36‘,
# } #对付一般的反爬措施
# r = requests.get(url, headers=headers)
# r.status_code #200

# 从爬到的东西提取内容（js太复杂 不考虑）  两个思路
# 方法一：通过字符串函数或正则表达式来寻找匹配要求的数据，难度较高难以掌握，且对网页适应性较差。
# 方法二（好）：利用网页的dom库结构，专用的网页解析库，如自带的HTMLParser、第三方的BeautifulSoup
# 或etree（一个 xpath解析库），特点是利用了网页的文档结构特点，可快速定位和提取数据。
# https://www.w3school.com.cn/htmldom/dom_nodes.asp  dom树

import requests
from bs4 import BeautifulSoup
url = 'https://news.seu.edu.cn/5487/list.htm' #东大头条
server = 'https://news.seu.edu.cn'#根目录 
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.117 Safari/537.36',
}
r = requests.get(url, headers=headers)
r.encoding = 'utf-8'
html = r.text
bs = BeautifulSoup(html, 'lxml')#lxml模型
for span in bs.find_all('div', class_='news_title'): #常用参数还有id，注意class_
    links = span.find('a')#找a标签  里面是相对路径加上根目录server
    if links:  #没有a标签则不打印
        print(links.text , server + links['href'])  #内容和属性

# xpath  etree  更灵活
from lxml import etree
#作为示例的 html文本
html = '''<div class="container">  
                              <a href="#123333" class="box">
                                    点击我
                                </a>            
</div>'''
#对 html文本进行处理 获得一个_Element对象
dom = etree.HTML(html)
#获取 a标签下的文本
a_text = dom.xpath('//div/a/text()')  #返回一个列表  div下的a下的文本
print(a_text[0].strip()) #去除空白字符
# a / b ：‘/’表示直接层级关系
# a // b：可以是间接子节点
# [@]：选择具有某个属性的节点
# //div[@class], //a[@x]：选择具有 class的 div节点、选择具有 x的 a节点 
# //div[@class="container"]：选择具有 class属性的值为 container的 div节点
# //a[contains(text(), "点")]：选择文本内容里含有 “点” 的 a标签
# //a[contains(@id, "abc")]：选择 id属性里有 abc的 a标签
# //a[contains(@y, "x")]：选择有 y属性且 y属性包含 x值的 a标签

import requests
from lxml import etree
url = 'https://news.seu.edu.cn/5487/list.htm' #东大头条
server = 'https://news.seu.edu.cn'#根目录 
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
}
r = requests.get(url, headers=headers)
r.encoding = 'utf-8'
html = r.text
dom = etree.HTML(html)
a_text = dom.xpath('//div[@class="news_title"]/a/text()')
a_url = dom.xpath('//div[@class="news_title"]/a/@href')
print(a_text,a_url)

# 爬图片
# url = 'http://www.netbian.com/'
# r = requests.get(url)
# r.encoding = r.apparent_encoding
# dom = etree.HTML(r.text)
# img_path = '//a[@title]/img'  #标签下的图像
# imgs = dom.xpath(img_path)
# for img in imgs:
#     src = img.xpath('@src')[0]  #出来是列表  要用0选第一个
#     name = img.xpath('@alt')[0] #文件名
#     image = requests.get(src)
#     with open(name+'.jpg', 'wb') as f:
#         f.write(image.content)

# # 简书 爬点赞评论数等
# url = 'https://www.jianshu.com/u/4ba5b6b1a57c'
# #第一种写法的 xpath 不好
# # xpath_link = '//ul[@class="note-list"]/li/div/a/@href'
# # xpath_title = '//ul[@class="note-list"]/li/div/a/text()'
# # xpath_comment_num = '//ul[@class="note-list"]/li/div/div[@class="meta"]/a[2]/text()'#第二个a标签
# # xpath_heart_num = '//ul[@class="note-list"]/li/div/div[@class="meta"]/span/text()'
# # #获取和解析网页
# # r = requests.get(url, headers=headers)
# # r.encoding = 'utf8'
# # dom = etree.HTML(r.text)
# # #所有的 链接 标题 评论数 点赞数
# # links = dom.xpath(xpath_link)
# # titles = dom.xpath(xpath_title)
# # comment_nums = dom.xpath(xpath_comment_num)
# # heart_nums = dom.xpath(xpath_heart_num)
# # #将每篇文章的链接 标题 评论数 点赞数放到一个字典里
# # data = []
# # for i in range(len(links)):
# #     t = {}
# #     t['link'] = links[i]
# #     t['title'] = titles[i]
# #     t['comment_num'] = comment_nums[i].strip()
# #     t['heart_num'] = heart_nums[i].strip()
# #     data.append(t)
# # #打印结果
# # for t in data:
# #     print(t)   #why???

# #第二种写法的 xpath  好
# #获取所有 li标签
# xpath_items = '//ul[@class="note-list"]/li'
# #对每个 li标签再提取
# xpath_link = './div/a/@href'
# xpath_title = './div/a/text()'
# xpath_comment_num = './/div[@class="meta"]/a[2]/text()'
# xpath_heart_num = './/div[@class="meta"]/span/text()'
# #获取和解析网页
# r = requests.get(url, headers=headers)
# r.encoding = r.apparent_encoding
# dom = etree.HTML(r.text)
# #获取所有的文章标签
# items = dom.xpath(xpath_items)
# #分别对每一个文章标签进行操作 将每篇文章的链接 标题 评论数 点赞数放到一个字典里
# data = []
# for article in items:
#     t = {}
#     t['link'] = article.xpath(xpath_link)[0]
#     t['title'] = article.xpath(xpath_title)[0]
#     t['comment_num'] = ''.join(article.xpath(xpath_comment_num)).strip()  #拼起来
#     t['heart_num'] = article.xpath(xpath_heart_num)[0].strip()   #再切开
#     data.append(t)
# #打印结果
# for t in data:
#     print(t)

# 正则表达式  模糊匹配字符串
# 它的设计思想是用一种描述性的语言来给字符串定义一个规则，凡是符合规则的字符串，
# 我们就认为它“匹配”了，否则，该字符串就是不合法的。
# Python通过re模块提供对正则表达式的支持。使用re的一般步骤是先将正则表达式
# 的字符串形式编译为Pattern实例，然后使用Pattern实例处理文本并获得匹配结果
# （一个Match实例），最后使用Match实例获得信息，进行其他的操作。

import re
regex = re.compile('abc')
content = 'abc'
y = regex.match(content)
print(y)
print(type(y))
print(y.group())
print(y.span())

#手机号码匹配
import re
regex = re.compile('1[3-8]\d{9}')  #re.compile('1[3-8]\d{9}$')
content = '18155825579'  #'18155825579abcd'
y = regex.match(content)
print(y)
# 注意：match方法对^不敏感，因为本身就是从左侧开始的

#单词匹配
import re
a = re.match(r'^\w+ve','hover')
b = re.match(r'^\w+ve\b','hover')
print(a)
print(b)
# 注意： \b 匹配这样的位置：它的前一个字符和后一个字符不全是(一个是,一个不是或不存在) \w。

a = re.match(r'^\w+\bve\b','hover')  
b  = re.match(r'^\w+\b\sve\b','hover ve') 
c = re.match(r'^\w+\bve\b','ho ve r')
d = re.match(r'^\w+\s\bve\b','ho ve r')
e = re.match(r'^\w+\bve\b','hove r')
f =  re.match(r'^\w+ve\b','hove r') 

import re
a = re.match(r'^.+ve\B','ho ve r')  
b = re.match(r'^.+ve\B','ho ver')  
c = re.match(r'^.+\Bve\B','ho ver')  
d = re.match(r'^.+\Bve\B','hover')

#匹配出0-100之间的数字
import re
r'[1-9]\d?$|0$|100$'

# 提取 <h1>网页标题</h1> 中的 网页标题
import re
result = re.match(r'<h1>(.*)</h1>','<h1>网页标题</h1>')  #变量保存
a = result.group()  #默认0，返回匹配内容
b = result.group(0) #输入0，返回匹配内容
c = result.group(1)  #返回第1个括号内容
d = result.groups()
print(a,b,c,d)

import re
result = re.match(r'<span>(\d+)</span><h1>(.*)</h1>','<span>1234</span><h1>网页标题</h1>')
a = result.group(1)
b = result.group(2)
c = result.groups()
print(a , b , c)

import re
s = '<html><h1>my web</h1></html>'
a = re.match(r'<.+><.+>.+</.+></.+>',s)   #关注html和h1这些内容， .+ 表示至少有内容
s = '<html><h1>my web</h1></h>'    #把s后面的html改成h，明显格式不对，但还是匹配了
b = re.match(r'<.+><.+>.+</.+></.+>',s)
print(a , b)

import re
s = '<html><h1>my csdn</h1></h>'          #错误的标签格式
a = re.match(r'<(.+)><(.+)>.+</\2></\1>',s)   #此时不符合了
s = '<html><h1>my csdn</h1></html>'       #正确的标签格式
b = result = re.match(r'<(.+)><(.+)>.+</\2></\1>',s)
print(a , b)

import re
s = '<html><h1>my csdn</h1></h>'    #不正确的格式
a = re.match(r'<(?P<key1>.+)><(?P<key2>.+)>.+</(?P=key2)></(?P=key1)>',s)
s = '<html><h1>my csdn</h1></html>'  #正确的格式
b = re.match(r'<(?P<key1>.+)><(?P<key2>.+)>.+</(?P=key2)></(?P=key1)>',s)
print(a, b)

# 数量词的贪婪模式与非贪婪模式：正则表达式通常用于在文本中查找匹配的字符串。总是尝试匹配尽可能多的字符；非贪婪的则相反，总是尝试匹配尽可能少的字符。例如：正则表达式"ab*"如果用于查找"abbbc"，将找到"abbb"。而如果使用非贪婪的数量词"ab*?"，将找到"a"。
# compile(pattern，flags= 0)	使用任何可选的标记来编译正则表达式的模式，然后返回一个正则表达式对象
# re.I(re.IGNORECASE): 忽略大小写（括号内是完整写法）
# re.M(MULTILINE): 多行模式，改变'^'和'$’的行为
# re.S(DOTALL): 点任意匹配模式，改变'.'的行为
# match(pattern，string，flags=0)	尝试使用带有可选的标记的正则表达式的模式来匹配字符串。如果匹配成功，就返回匹配对象； 如果失败，就返回 None
# search(pattern，string，flags=0)	使用可选标记搜索字符串中第一次出现的正则表达式模式。 如果匹配成功，则返回匹配对象； 如果失败，则返回 None
# findall(pattern，string[, flags] )	查找字符串中所有（非重复）出现的正则表达式模式，并返回一个匹配列表
# sub(pattern，repl，string，count=0)	使用 repl 替换所有正则表达式的模式在字符串中出现的位置，除非定义 count， 否则就将替换所有出现的位置
# split(pattern，string，max=0)	根据正则表达式的模式分隔符， split函数将字符串分割为列表，然后返回成功匹配的列表，分隔最多操作 max 次（默认分割所有匹配成功的位置）

result = re.search(r'\d+','文章篇幅为1314521个字')
result1 = re.search(r'^\d+','文章篇幅为1314521个字')
result2 = re.findall(r'\d+','文章篇幅为1314521个字,共234页,有1000张图')

import re
s = '''
<html>
<p>python开发工程师</p>
<p><span>+ python爬虫工程师</span></p>
</html>
'''
re.sub(r'<.+>','',s)  #结果为什么不对？
# re.sub(r'</?\w+/?>','',s)

s = "language:python,php,c,cpp-java"
re.split(r':|,|-',s)

# 豆瓣爬电影 爬虫圣地
import requests
import re
import json
def parse_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0"}
    response = requests.get(url, headers=headers)
    text = response.text
    regix = '<div class="pic">.*?<em class="">(.*?)</em>.*?<img.*?src="(.*?)" class="">.*?div class="info.*?class="hd".*?class="title">(.*?)</span>.*?class="other">' \
            '(.*?)</span>.*?<div class="bd">.*?<p class="">(.*?)<br>(.*?)</p>.*?class="star.*?<span class="(.*?)"></span>.*?' \
            'span class="rating_num".*?average">(.*?)</span>'
    # .*？ 任意字符  0-任意个  非贪婪
    results = re.findall(regix, text, re.S)
    for item in results:
        down_image(item[1],headers = headers)
        yield {
            '电影名称' : item[2] + ' ' + re.sub('&nbsp;','',item[3]),
            '导演和演员' : re.sub('&nbsp;','',item[4].strip()),
            '评分': star_transfor(item[6].strip()) + '/' + item[7] + '分',
            '排名' : item[0]
        }
def main():
    for offset in range(0, 250, 25):
        url = 'https://movie.douban.com/top250?start=' + str(offset) +'&filter='
        for item in parse_html(url):
            print(item)
            write_movies_file(item)
def write_movies_file(str):
    with open('douban_film.txt','a',encoding='utf-8') as f:
        f.write(json.dumps(str,ensure_ascii=False) + '\n')

def down_image(url,headers):
    r = requests.get(url,headers = headers)
    filename = re.search('/public/(.*?)$',url,re.S).group(1)
    with open(filename,'wb') as f:
        f.write(r.content)
def star_transfor(str):
    if str == 'rating5-t':
        return '五星'
    elif str == 'rating45-t' :
        return '四星半'
    elif str == 'rating4-t':
        return '四星'
    elif str == 'rating35-t' :
        return '三星半'
    elif str == 'rating3-t':
        return '三星'
    elif str == 'rating25-t':
        return '两星半'
    elif str == 'rating2-t':
        return '两星'
    elif str == 'rating15-t':
        return '一星半'
    elif str == 'rating1-t':
        return '一星'
    else:
        return '无星'
if __name__ == '__main__':
    main()

# post 和服务器交互 例如注册
# import requests
# # 'http://47.98.171.106/python/test1.html
# url = 'http://47.98.171.106/python/test.php'
# data = {
#     'username': ‘seu',
#     'password': '123456',
# }
# r = requests.post(url, data=data)
# print(r.status_code)
# print(r.text)

# import requests
# url = 'http://47.98.171.106/python/test.php'
# session = requests.session()
# #构造 data 字典 username 、password 分别和上方网页中的 username 和 password 对应
# data = {
#     'username': 'seu',
#     'password': '123456',
# }
# session.post(url, data=data)
# url = 'http://47.98.171.106/python/admin.php'
# response=session.get(url=url)
# print(response.text)

# 网站道德约束robots.txt  百度不能访问淘宝