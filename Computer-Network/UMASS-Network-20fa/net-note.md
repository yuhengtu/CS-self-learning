# 1.1 intro

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312091759913.png" alt="image-20231209175956768" style="zoom: 50%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312091802315.png" alt="image-20231209180224176" style="zoom:90%;" />



# 1.2 Network Edge (Access networks, physical media)

access network，考虑speed 和 share程度(shared/dedicated)

![image-20231209225952730](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092259785.png)

FDM: diff freq互不干扰，shared freq，standard: docsis，ch6

![image-20231209230210031](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092302126.png)

更多是接受数据而不产生数据，downstream快；shared

![image-20231209230511665](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092305714.png)

电话线，3miles以上就不行了

![image-20231209231028444](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092310521.png)

![image-20231209231203502](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092312567.png)

wireless，ch7

![image-20231209231532692](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092315724.png)

![image-20231209231542188](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092315222.png)![image-20231209231604879](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092316914.png)

![image-20231209231716380](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092317435.png)

![image-20231209231759659](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092317780.png)



packet

![image-20231209232453551](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092324610.png)



physical media: copper wires, fiber optics, radio links

![image-20231209232709683](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092327736.png)

coaxial已经淘汰，fiber很贵

![image-20231209232919842](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092329898.png)

![image-20231209233311711](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312092333773.png)

# 1.3 Network Core (Forwarding, routing; packet/curcuit switching; Internet)

![image-20231211152932637](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111529742.png)

local forwarding (本地进出) & global routing (总体路线)

![image-20231211153118569](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111531601.png)

![image-20231211153423923](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111534968.png)



1、packet switching

完整收集到整个packet才会继续传送

![image-20231211153610417](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111536458.png)

queue memory 不够就会 lost

![image-20231211155625531](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111556570.png)

2、互联网出现之前，电话线采用 circuit switching

![image-20231211154820421](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111548460.png)

 ![image-20231211154917983](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111549029.png)

二者比较，packet switching 容忍极小丢包率，增加用户数，有 statistical multiplexing gain 统计复用增益

![image-20231211155604840](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111556882.png)

![image-20231211160014913](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111600961.png)



network of networks

![image-20231211160420607](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111604665.png)

![image-20231211160521829](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111605870.png)

![image-20231211160634291](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111606329.png)

Content provider 常绕过  tier-1/ISP

![image-20231211160844376](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111608421.png)

![image-20231211161159794](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111611851.png)

![image-20231211161249235](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312111612295.png)

# 1.4 Performance (Delay, Loss, Throughput )

4种delay

![image-20231212235340789](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312122353860.png)

![image-20231212235354394](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312122353416.png)

![image-20231212235434388](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312122354409.png)

![image-20231212235454350](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312122354370.png)

![image-20231212235553897](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312122355918.png)

 区分 trans 和 prop，类比

![image-20231213000431132](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130004159.png)

![image-20231213000735471](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130007498.png)

延迟测试：先发3个packet 到第一个hop router；round trip time: rtt ，发送到接收的时间差；再发第二、第三...个hop router

![image-20231213001036125](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130010156.png)

![image-20231213001316199](../Library/Application Support/typora-user-images/image-20231213001316199.png)

  *** 通常表示拒绝reply

![image-20231213001551553](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130015580.png)

loss, buffer/queue

![image-20231213001754711](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130017736.png)

类比水管

![image-20231213001945871](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130019896.png)

throughput，被链路中最细的水管限制

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130022166.png)

每条链路经过三个pipe

![image-20231213002450749](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312130024774.png)



# 1.5 layering, encapulation, service model

layering 类比

![image-20231214101230483](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141012834.png)

![image-20231214101437849](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141014873.png)

Transport layer, 不丢包TCP，可丢包UDP

network layer: host to host transfer (end device), not reliable (best effort service); transport layer: process to process transfer

![image-20231214101923450](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141019472.png)

![image-20231214102522835](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141025855.png)

encapulation封装：把高层的message，加上一些信息，变成低层的message

![image-20231214102757032](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141027054.png)

![image-20231214102854818](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141028839.png)

![image-20231214102946073](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141029098.png)

![image-20231214103102057](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141031079.png)

![image-20231214103306282](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141033309.png)

# 1.6 Networks under attack

![image-20231214104221545](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141042573.png)

bad guy 抓包

![image-20231214104323560](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141043586.png)

钓鱼邮件，src假装是B

![image-20231214104511564](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141045586.png)

![image-20231214104730542](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141047568.png)

![image-20231214105004330](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141050359.png)

# 1.7 History of Networking

![image-20231214111053131](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141110154.png)

三人同时独立发明 packet swiching；NCP是tcp/ip的祖先 

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141108584.png)

![image-20231214112015716](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141120745.png)

![image-20231214112243973](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141122002.png)

![image-20231214112527872](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141125904.png)

![image-20231214112726267](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141127292.png)

# 1.8 Who use/control the Internet?

a4ai.org statics

Speedtest.net

![image-20231214114835869](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141148890.png)

![image-20231214114900200](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141149221.png)

![image-20231214115126111](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141151135.png)

![image-20231214115137579](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141151607.png)

Infrastructure layer

![image-20231214115729253](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141157285.png)

![image-20231214115821558](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141158583.png)

![image-20231214115858637](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141158661.png)

Names, numbers

网址冲突可以发起争议，网址可以卖钱

![image-20231214120429580](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141204608.png)

![image-20231214120724011](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141207038.png)

content layer

![image-20231214121017560](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141210594.png)

![image-20231214121222294](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141212329.png)

![image-20231214121256575](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141212601.png)

防火墙，禁止访问某域名/检查包内不包含x

![image-20231214121500600](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312141215628.png)

# 2.1 Application layer (client-server, P2P, sockets, transport tcp/udp)

断断续续连接，动态ip；http, client: web browser, server: web server

![image-20231216150620678](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161506774.png)

peers之间直接联系，每个peer都有client/server的功能

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161508790.png)

executing program -> process，计算机内进程通信是 IPC(inter-process communication)，计算机间通信是message

![image-20231216152113547](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161521582.png)

socket类比为door

![image-20231216152350999](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161523027.png)

找到对应的socket要门牌号，即ip: port

![image-20231216173546357](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161735406.png)

protocol格式分公开和公司私有

![image-20231216173926981](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161739015.png)

不同类型数据有不同追求

![image-20231216174612763](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161746801.png)

![image-20231216174705367](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161747404.png)

transportation layer: tcp & udp; can build services udp not provide on top of udp in applicaiton layer 

![image-20231216175105007](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161751037.png)

![image-20231216175047677](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161750722.png)

tcp/udp没有security设计，需要在app layer实现，如TLS

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161753926.png" alt="image-20231216175333883" style="zoom:25%;" />

# 2.2 HTTP: (non)persistent, messages, cookies, caching, http2/3

web page = base html file + objects

![image-20231216182829037](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161828081.png)

![image-20231216183020904](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161830941.png)

stateless，只有收到request然后reply

![image-20231216183340865](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161833907.png)

non- persistent需要一个rtt来建立tcp，另一个rtt来request&response；persistent http即http 1.1

![image-20231216183736144](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161837183.png)

![image-20231216183957358](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161839400.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161847931.png" alt="image-20231216184744885" style="zoom:60%;" />

![image-20231216184809053](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161848090.png)

![image-20231216185115578](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161851624.png)

HTTP message

![image-20231216185731706](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161857759.png)

GET没有entity body，POST有

![image-20231216185949434](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161859468.png)

![image-20231216190155454](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161901510.png)

status code 200, status phrase OK

![image-20231216190345435](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161903499.png)

![image-20231216190547428](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161905484.png)

HTTP is stateless, cookies keep past state

![image-20231216190750483](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161907521.png)

client 原本有ebay cookies；第一次向amazon server发送request，amazon把信息存入database，reply包含cookies，

![image-20231216191453988](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161914031.png)

gdpr法规

![image-20231216191639193](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161916244.png)

第一种cache，Web cache

![image-20231216180234914](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161802966.png)

![image-20231216180411840](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161804878.png)

15 * 100k = 1.5M, 1.5 / 1.54 = 97%; suppose institutional network -> client is 1Gbps Ethernet, utilization = 0.0015

由于0.97，access link delay由于queue delay会达到分钟级别，其他都很少

![image-20231216182041256](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161820313.png)

![image-20231213000735471](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161813556.png)

![image-20231216182453258](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161824310.png)

第二种cache，conditional get，在client本地保存一份copy，如果是最新的，not modified，就不用重传

![image-20231216192049579](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161920624.png)

Http2 & 3

减轻head of the line blocking

![image-20231216192810637](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161928682.png)

HOL blocking

![image-20231216192902541](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161929594.png)

![image-20231216192956097](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161929136.png)

![image-20231216193025524](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312161930561.png)

# 2.3 Email: SMTP, IMAP

![image-20231217013858079](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170138172.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170149156.png)

![image-20231217015429578](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170154671.png)

![image-20231217015556375](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170155441.png)

Alice: client; Bob: server；一个单一句点表示结束

![image-20231217015913014](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170159066.png)

HTTP，拉取信息 server2client；SMTP，推送信息 client2server；persistent，一次连接发送多条email

![image-20231217020238605](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170202653.png)

![image-20231217020629713](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170206765.png)

SMTP 发邮件；IMAP 收邮件

![image-20231217020929375](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170209494.png)

# 2.4 DNS: Domain Name Service

Translate host name into IP address (32bit); application layer level

domain 域名；www.example.com，".com" 是顶级域（TLD），"example" 是二级域名，"www" 是三级域名。

DNS工作在network edge; notice the philosophy of "complexity" at edge and "simplicity" at core

![image-20231217021648862](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170216924.png)

Aliasing 别名，external hostname转换为更复杂的internal hostname；load balancing function负载均衡，有很多ip server都可以提供同样的服务，从中进行选择

DNS的请求太多，因此采用decentralized/distributed

![image-20231217023020278](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170230338.png)

record simple 但很多

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312170233621.png" alt="image-20231217023303562" style="zoom:25%;" />

root -> TLD -> authoritative

![image-20231217150648978](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171506067.png)

root

![image-20231217151141310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171511357.png)

TLD & authoritative

![image-20231217151255314](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171512379.png)

![image-20231217151601560](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171516605.png)

Iterated query: 1246 是query，35reply接下来去找哪个server问，7reply了最终ip

![image-20231217151835310](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171519879.png)

Recursive query，把压力给到server，因此不常用；常用Iterated query

![image-20231217152225428](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171522500.png)

caching，TTL time to live；best- effort approach: 不实时检测是否有更新，等ttl结束后自然会被更新，容忍短时间的错误

![image-20231217153101101](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171531174.png)

DNS record (RR )：typeA用于hostname ->  ip translate；NS nameserver record；Cname用于 name aliasing；MS give the name of a mail server 

![image-20231217153726253](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171537322.png)

DNS protocal message，query和对应的reply使用同一个id

![image-20231217154413345](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171544422.png)

![image-20231217154530393](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171545468.png)



![image-20231217154930319](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171549396.png)

![image-20231217155034570](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171550646.png)

# 2.5 Peer-to-Peer File Distribution

![image-20231217174613489](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171746628.png)

![image-20231217174720940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171747022.png)

![image-20231217174731712](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171747789.png)

![image-20231217175233380](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171752463.png)

![image-20231217175408613](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171754691.png)

BitTorrent 是一种 P2P 文件分发协议，用于高效地共享和分发大文件

![image-20231217175421163](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171754237.png)

![image-20231217175924160](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171759220.png)

choke 掐死

![image-20231217180241199](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171802260.png)

![image-20231217180314768](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312171803850.png)

# 2.6 Video Streaming and Content Distribution Networks

app layer infrastructure

ISP（Internet Service Provider）：ISP指的是互联网服务提供商

CDN（Content Delivery Network）： CDN是一种通过分布在全球不同位置的服务器来加速互联网上内容传递的网络架构。

![image-20231218115301758](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181153870.png)

24 frame/s

![image-20231218115636359](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181156422.png)

2种video coding 

![image-20231218115850236](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181158298.png)

![image-20231218120047588](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181200660.png)

3steps: record, sent, play at client

![image-20231218120429247](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181204308.png)

Jitter 抖动，解决变化延迟

![image-20231218121252401](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181212463.png)

DASH，解决variable bandwidth；manifest清单，上面记录了存储不同码率不同文件的CDN nodes；选择最大的可支持码率

![image-20231218140737285](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181407448.png)

![image-20231218141508349](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181415445.png)

Point of Presence (PoP): 在网络中提供连接服务的地理位置

![image-20231218141921611](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181419712.png)

很多CDN上都有MADMEN电影，在家想看MADMEN电影，Netflix软件向server（最右侧）发出请求，server返回manifest（存有video chunks and locations），之后从邻近CDN获取电影若congest则换一个CDN

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181429791.png)

netflix是OTT，over-the-top service，在app layer，使用ip infrastructure

![image-20231218143433169](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312181434261.png)

# 2.7 Socket Programming: UDP/TCP

socket is the only API between app layer & transport layer

![image-20231219194312466](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312191943247.png)



![image-20231219194533663](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312191945696.png)

UDP: unreliable, unordered, no connection

![image-20231219195556480](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312191955543.png)

Python: AF_INET 表示Internet type socket IPv4, SOCK_DGRAM 表示UDP socket

![image-20231219200344293](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192003331.png)

![image-20231219200417828](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192004865.png)

用bind指定port number

![image-20231219200554406](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192005444.png)

TCP: first server welcome socket, second server create new socket for future contact with specific client; the new socket protal num = welcome socket

![image-20231219200920784](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192009821.png)

 connection setup 信息是在transport layer发的；connection socket is the new socket for future contact with specific client

![image-20231219202450507](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192024557.png)

SOCK_STREAM表示tcp，connection establish之后无需指定name&port 

![image-20231219202615110](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192026166.png)

创建并使用connection socket

![image-20231219203201970](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192032024.png)

![image-20231219203932693](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192039736.png)

![image-20231219205123403](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192051444.png)

# 2.8 3rd party cookies, GDPR

http is stateless, cookies help remember user habit, also can be used to track through multiple website

advertisement from other server

![image-20231219205719635](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192057682.png)

![image-20231219205958450](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192059486.png)

![image-20231219210734728](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192107783.png)

how cookies be used; cookie jar (database); cookie包含语言偏好，账号密码等

![image-20231219211327965](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192113017.png)

![image-20231219211550758](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192115807.png)

Adx的cookie可以知道我访问过哪些网站，并且很多网站都有Adx的广告，他会开始推送我看过的东西；向Adx发送GET时会包含referer和cookie

![image-20231219211911112](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192119158.png)

一天后再看nytimes

![image-20231219212144051](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192121097.png)

many third-party cookies are invisible (1 pixel image)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192126132.png)

![image-20231219213006545](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192130590.png)



# 3.1 Transport layer

![image-20231219213806683](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192138724.png)

logically communication意味着不考虑低层，抽象成一条直接连接的链路，可能lose/reorder/flip bits in message s

![image-20231219214019827](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192140865.png)

host类比为房子，process类比为房子中的children，message来的时候，又房子根据地址分发给children；房子中的message分发是transport layer，network layer类比为post service，负责送信

![image-20231219214808395](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192148431.png)

app message进入socket就是transport layer，tansportlayer加上Th，向下给到network layer，network layer在host间传送

![image-20231219215431482](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192154513.png)

![image-20231219220422972](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192204018.png)

流量控制(Flow Control)用于确保在数据的发送方和接收方之间维持适当的数据流量，防止接收方被过多的数据淹没而无法处理，适应接收方的处理能力

![image-20231219221346099](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192213138.png)



# 3.2 multiplexing and demultiplexing on TCP/UDP

multiplexing and demultiplexing 是遍布于各层中的idea

![image-20231219222411731](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192224770.png)![image-20231219222424943](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192224988.png)

重点在demultiplex到正确的地方

http service server 来的message，client如何demultiplex to firefox 而不是其他软件

两个client的firefox message在server上demultiplex到两个process

![image-20231219222631455](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192226502.png)

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192231558.png)



![image-20231219223300359](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192233414.png)

UDP demultiplex: 区分host-local port # & destination port #，UDP demultiplex只看destination port #

![image-20231219223508592](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192235649.png)

如图的port #都是进程号

![image-20231219224748140](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312192247195.png)

TCP demultiplex

![image-20231220152526881](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312201525947.png)

B的port都是80，HTTP server把他们 demultiplex 到Process4，P5，P6对应的socket

![image-20231220153004714](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312201530760.png)

![image-20231220153838450](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312201538494.png)

# 3.3 UDP, checksum

UDP 快，不可靠，面对congestion仍可工作（tcp不可）

![image-20231223005923276](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230059364.png)

DNS和SNMP，网络拥塞时也得工作(no congestion control)

![image-20231223010122096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230101135.png)

最简单易读的protocol

![image-20231223010355016](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230103054.png)

UDP sender&receiver

![image-20231223014155158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230141210.png)

![image-20231223014221355](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230142398.png)

![image-20231223014402897](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230144943.png)

checksum 校验，check header field和source&dest IP address，反码

![image-20231223014549983](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230145034.png)

![image-20231223014649309](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230146380.png)

最高位1需要再加回去，sum -> checksum 每一位翻转 

![image-20231223015009064](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230150130.png)

![image-20231223015315071](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312230153128.png)

 

# 3.4 RDT reliable data transfer, Pipelining (Go-back-N/Selective Repeat)

app layer单向可靠传输，建立在trans layer双向不可靠传输基础上，两边都互相看不到对方，且channel不可靠

![image-20231223134328185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231343283.png)

API

![image-20231223134438783](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231344819.png)

rdt protocol

![image-20231223134743918](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231347988.png)

类比，灯泡有on & off state

![image-20231223135144116](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231351152.png)

sender & receiver分别有state

先假设channel reliable

![image-20231223135544130](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231355177.png)

channel unreliable

![image-20231223135832444](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231358490.png)

![image-20231223141445766](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231414842.png)

![image-20231223141654929](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231416976.png)

2.1，receiver自动拒收重复packet（连续两次同sequence的）

![image-20231223142004542](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231420598.png)

sender， sequence 0, 1；出现错误，sender to receiver packet出错/AK NAK回传出错；AK NAK回传出错，会再发一次，原本是sequence 0 则重发sequence 0，原本是sequence 1 则重发sequence 1

![image-20231223143138585](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231431631.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231436364.png" alt="image-20231223143632276" style="zoom:50%;" />

![image-20231223143856906](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231438948.png)

2.2没有使用NAK

![image-20231223144010193](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231440234.png)

![image-20231223145428159](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231454234.png)

目前解决了pkt corrupt问题，还没有解决pkt loss 问题

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231459332.png)

![image-20231223150901666](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231509720.png)

![image-20231223161318466](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231613580.png)

![image-20231223161406984](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231614029.png)

![image-20231223161938011](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231619092.png)

蓝色区域是第一个pkt到最后一个pkt的间隙 = L/R

![image-20231223161947425](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231619496.png)

表现很差，protocol 限制了1GB link

![image-20231223162046444](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231620512.png)

引入pipeline，提升performance

![image-20231223162356587](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231623660.png)

3 pipelining使得utilization翻了3倍

![image-20231223162509584](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231625656.png)

2种pipeline方案：Go-Back-N & Selective repeat；

Go-Back-N只有一个timer；cumulative积累的；收到seq n 的ACK，就ACK所有n及其之前的pkt（TCP基于此），timeout会重传所有n及其之后的pkt

![ ](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231627662.png)

sender只发送seq #最大的ACK，in-order 按顺序的；遇到乱序的会discard/buffer

![image-20231223163509185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231635247.png)

![image-20231223170937368](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231709466.png)

 Selective repeat： 每个pkt有独立的timer，独立ACK；缓冲乱序的pkt，等都到了再按顺序向上层传送

![image-20231223171344620](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231713668.png)

![image-20231223171914792](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231719865.png)

sender，如果n是最小的unACKed packet，被ACK后，window move forward；receiver重发ACK以免之前的ACK已丢失

![image-20231223172053101](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231720151.png)

![image-20231223172534315](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312231725365.png)



# 4.1 Network layer

之前app和trans layer都是network edge，之后进入network core

![image-20231224022331861](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240223956.png)

![image-20231224022432960](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240224032.png)

data/control plane，分别是forwarding / routing

![image-20231224022723012](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240227085.png)

compute出这个local forwarding table有两种算法，即traditional routing algorithms / SDN

![image-20231224022902735](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240229817.png)

![image-20231224023154237](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240231319.png)

IP, best effort service

![image-20231224023742714](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240237793.png)

历史上其他科研尝试，1990s

![image-20231224023950227](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312240239299.png)

![image-20231224024117596](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241346508.png)

# 4.2 Router: Input/output ports, switching fabrics, Network Neutrality

![image-20231224134755543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241347592.png)

input port；红色是network layer；lookup + forwarding, match + action；generalized forwarding, determine on network/link/physics layer header, 如TCP/UDP去到两个不同的output port

![image-20231224135316637](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241353693.png)

![image-20231224135327722](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241353763.png)

longest prefix matching

![image-20231224143001901](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241430997.png)

同时满足1和2，选择1

![image-20231224143036305](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241430383.png)

longest prefix matching是硬件实现的(TCAMs)，非常快，不论多大table一个时钟周期

![image-20231224143423829](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241434908.png)

switching rate: max rate from input port to output port; N inputs, R line rate, switching rate = NR -> non-bloacking switch 很贵

![image-20231224143618646](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241436733.png)

更好的做法是tolerate some block，在input port的红色框内queue，input queueing；第三种是最常用的

![image-20231224144648436](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241446519.png)

看作一种IO device

![image-20231224144942970](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241449051.png)

![image-20231224145019396](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241450486.png)

Clost network, use parallelism

![image-20231224145153917](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241451009.png)

Cisco公司

![image-20231224145605674](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241456728.png)



Output port: buffering management/pkt scheduling

HOL: input port中很多红色pkt都想去红色output port，有些红色input pkt因此queue，也堵住了后面绿色pkt；这时绿色pkt遭遇HOL

![image-20231224150606533](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241506593.png)

假设non-bloacking switch，fabric NR，output port R -> buffering；drop/schedule policy，drop谁，先transmit谁 

![image-20231224151346016](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241513081.png)

how many buffering这个问题仍然是争议；rule of thumb 经验法则；large buffer -> less loss, large delay/RTT, large RTT导致TCP detect/react to congestion slower

![image-20231224152118429](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241521493.png)

buffering management

drop有两种policy，第一种后来的drop；第二种按优先级drop，可drop正在queue的；ECN, 2bit in IP header are set when congestion

![image-20231224152849884](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241528945.png)

![image-20231224153600543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241536608.png)

头等舱

![image-20231224153735228](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241537317.png)

三个class轮流发送1pkt，如此循环

![image-20231224154201108](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241542182.png)

![image-20231224154356715](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241543772.png)

Network Neutrality

![image-20231224155014869](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241550929.png)

ISP干扰Bittorrent事件; company cannot pay to get faster service

![image-20231224155049111](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241550195.png)

![image-20231224155924365](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241559499.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312241600088.png" alt="image-20231224160044982" style="zoom: 25%;" />

# 4.3 The Internet Protocol (IP): IPv4, addressing, NAT, IPv6

IP protocol 并不是routing algorithm；pkt handing 指的是如何拆解成pkt

![image-20231225101650252](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251016361.png)

大多datagram没有options，header范围到destination IP address，total范围指header + payload (变长，之后向上传给trans layer)；ECN，congestion notification；diffserv 区分 service class（4.2的调度算法）；TTL到0就drop；fragmentation/reassembly 三个用来区分不同pkt，IPv6中没有；checksum由header计算，TTL会随时间变化，checksum要在每个router计算，因此IPv6也删除了

![image-20231225102438324](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251024392.png)

32bit，每8bit用.分割

![image-20231225104318951](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251043041.png)

![image-20231225104749208](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251047279.png)

同一个subnet 不同的host：高位同，低位不同

![image-20231225104914463](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251049534.png)

高24bit 相同 -> /24 subnet

![image-20231225105029385](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251050489.png)

去掉router，可见有6个subset

![image-20231225105656276](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251056344.png)

CIDR notation 

![image-20231225105726728](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251057817.png)

zero configuration/plug-and-play 指通过DHCP协议获取IP

![image-20231225110204597](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251102670.png)

一个设备leave后他的IP会被分配给之后的设备

![image-20231225110808281](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251108354.png)

![image-20231225112443603](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251124736.png)

![image-20231225113157539](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251131610.png)

![image-20231225113212266](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251132336.png)

DHCP是UDP，client port 67，server port 68；client还没有ip，因此src：0.0.0.0，广播查找DHCP server，因此des：255.255.255.255；yiaddr：your  internet address；transaction ID用来对应确认；DHCP request可以是刚offer的IP，或者想要继续用之前的IP（这种情况省略前两步DHCP discover/offer）

![image-20231225115226207](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251152334.png)

DCHP server的其他功能：host 需要知道 first-router address，即第一个发给哪个router；network mask 指区分subnet/host part

![image-20231225115509941](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251155019.png)

ISP是/20，organization是/23，3bit -> 分配给8个client

![image-20231225145800212](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251458375.png)

parent ISP: Fly-By-Night-ISP; 8 client

![image-20231225154334043](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251543205.png)

/20 -> /23, longest prefix match

![image-20231225154347672](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251543739.png)

更高一层，ISP如何获取IP地址范围？；NAT network address translation

![image-20231225154602746](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312251546829.png)



NAT

![image-20231225221452332](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252214402.png)

![image-20231225221749631](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252217670.png)

![image-20231225222222860](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252222918.png)

向外发送，S转换，D不变；向内接收，D转换，S不变

![image-20231225222339550](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252223588.png)

![image-20231225222709418](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252227460.png)



IPv6，以flow为对象，代替datagram

![image-20231225222952939](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252229980.png)

16bit flow label; flow handling policy由ISP决定；pri是 service class（4.2的调度算法）；去除了三个部分，fixed-length format，fast processing

![image-20231225223139182](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252231237.png)

有 IPv6，IPv4，IPv6/4 三种router

tunneling，IPv6作为IPv4的payload，IPv6/4做中间人

![image-20231225223805253](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252238308.png)

IPv4可以看做一种link layer service

![image-20231225224531163](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252245221.png)

![image-20231225224313545](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252243607.png)

NAT一定程度缓解了IPv4紧张地址问题；app layer的更新发展比network layer快得多

![](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312252248988.png)



# 4.4 Generalized fowarding, OpenFlow

近年Generalized fowarding正在替代router；many header: link layer frame header, network layer datagram header, trans layer segment header

![image-20231226014824647](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260148726.png)

Drop/block -> firewall; modify -> NAT; send to SDN controller；*表示忽略

![image-20231226015405052](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260154094.png)

OpenFlow规范；IP proto表示上层protocol，ToS type of service

![image-20231226015515173](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260155212.png)

port22用来ssh

![image-20231226015855088](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260158135.png)

![image-20231226020040171](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260200224.png)

layer3 routing; layer2 switching; NAT; Firewall

![image-20231226020109718](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260201758.png)

use SDN controller generate forwarding table for all client in network, no need for routing protocol

![image-20231226020426720](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260204765.png)

![image-20231226020648764](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260206807.png)

这是一种简单的program形式，只是在独立router内program pkt的行为; P4 language

![image-20231226020752104](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260207146.png)

# 4.5 Middleboxes

standard functions of an IP router -> des based forwarding of ip datagram; Middleboxes: at data path between hosts, at network core, at network layer

![image-20231226021501944](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260215421.png)

Load balancers: route 同一类 request to one of mirrored server, layer7/app layer switch

![image-20231226022128886](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260221936.png)

![image-20231226022228299](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260222343.png)

以上这些内容在丰富IP的作用

![image-20231226022620482](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260226530.png)

发福

![image-20231226022811536](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260228596.png)

![image-20231226022932452](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260229500.png)

intelligence, complexity at network edge, 即The end-end argument

![image-20231226023131566](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260231613.png)

![image-20231226023258574](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260233905.png)

![image-20231226023808156](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312260238210.png)

# 5.1 Network-layer control plane intro

dijkstra(centralized)/bellman(distributed); OSPF: intra domain(domain内部); BGP: inter domain(domain之间)

![image-20231230170624070](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312301706176.png)

2种control plane routing，其使用的算法都一样，只是整体架构不同

![image-20231230171452812](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312301714862.png)

![image-20231230171513707](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312301715742.png)



# 5.2 Link-State (LS) & distance-vector (DV) routing algorithm

![image-20231230194055354](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312301940451.png)

![image-20231230213812870](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302138973.png)

global/link state: 每个router都有所有router的连接&link cost; decentralized: 只有相接节点的link cost

![image-20231230214717975](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302147023.png)

![image-20231230215014940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302150987.png)



![image-20231230220948385](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302209492.png)

![image-20231230215540437](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302155483.png)

![image-20231230221925896](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302219946.png)

![image-20231230222025365](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302220403.png)

![image-20231230222157059](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302221118.png)

when link cost is dynamic; e is epsilon <1; define link cost based on short-term congestion/load level会出现震荡，因此后来不这样定义 link cost



Bellman-Ford (BF)

![image-20231230223559922](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302235984.png)

![image-20231230230200215](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302302285.png)

![image-20231230230253441](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302302479.png)

![image-20231230230650179](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302306228.png)

![image-20231230230858236](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302308286.png)

All nodes:

receive distance vectors from neighbors

compute their new local distance vector

send their new local distance vector to neighbors

![image-20231230231133106](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302311159.png)

![image-20231230231427415](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302314466.png)

t = 2时，ae节点才接收到c节点t = 0时的信息，像人类看见很久以前的星星

![image-20231230231606585](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302316660.png)

link cost 从4变成1，good news travels fast

![image-20231230231914183](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302319259.png)

link cost 从4变成60，bad news travels slow

![image-20231230232439970](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302324042.png)

LS failure is rather localized; DV可能有某个节点自称到所有地方都很近（黑洞），如一个小isp宣称自己连接到某大isp的路径cost = 0，所有pkt都想走小isp，实则他并不连接到大isp，也就吸走了所有pkt

![image-20231230235215047](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202312302352154.png)

# 5.3 Intra-AS Routing: OSPF link-state protocol

OSPF: within network; BGP: among network

scalable routing 可拓展

![image-20240102160614507](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021606602.png)

![image-20240102160823433](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021608476.png)

![image-20240102161152911](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021611954.png)

先看Intra，OSPF是基于dijkstra 最常用

![image-20240102161613607](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021616671.png)

malicious 恶意的；flood 广播

![image-20240102161624870](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021616904.png)

area border routers: summarize 到达内部area router的情况, advertise to other backbone router

![image-20240102161956754](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021619801.png)



# 5.4 Inter-AS Routing: BGP(Border Gateway Protocol)

BGP: glue the networks together; 基于DV算法；policy：不允许通过某个isp/country

![image-20240102165153602](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021651690.png)

e是AS之间，i是AS内部；gateway routers run both eBGP and iBGP protocols

![image-20240102165813774](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021658886.png)

BGP peers/speakers; BGP peers exchange over semi-permanent TCP at port 179; advertise path; AS3告诉AS2它可以reach X

![image-20240102170619533](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021706591.png)

![image-20240102170746281](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021707333.png)

advertise pat: prefix + attributes(属性: AS-PATH & NEXT-HOP); policy第二条，只要不告诉neighbor AS path，neighbor AS就永远不会通过那条path回传消息

![image-20240102170938185](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021709247.png)

example

![image-20240102191023697](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021910827.png)

chooses path *AS3,X*

![image-20240102191102000](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021911062.png)

policy; w是A的customer, y是C的customer, x是B和C的customer,; B不会把BAw这条路广播出去告诉C

![image-20240102192006580](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021920642.png)

![image-20240102193145436](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021931524.png)

forwarding tables

![image-20240102193813912](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021938970.png)

![image-20240102194045205](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021940259.png)![image-20240102194053343](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021940416.png)

Hot potato routing 烫手山芋，赶紧快速滚出本network

![image-20240102194200842](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021942889.png)

![image-20240102194941676](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202401021949755.png)

# 5.5 SDN Control Plane



# 5.6 ICMP: The Internet Control Message Protocol

ICMP: error/ping/traceroute





















# 5.7 Network Management and SNMP, NETCONF/YANG

