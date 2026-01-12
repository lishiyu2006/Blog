## 常用css属性缩写
### 对于和盒子四边相关的模型（margin和padding等）遵循时钟模型，即顺时针，上->左->右->下
- 4个值：上 左 右 下
- 3个值：上 左右 下
- 2个值：上下 左右
- 1个值：上左右下
### 其他
#### #border边框
3个属性：border-width，border-style,border-color
说明：三个属性可以随意调整位置
示例：
~~~css
/*边框宽度 边框类型 边框颜色*/
border: 1px soild black;
~~~

#### #border-radius边框圆角
4个属性：border-top-left-radius,border-top-right-radius,border-bottom-right-radius,border-bottom-left-radius
说明：类似于时钟模型，但是方向是左上->右上->右下->左下
示例1
~~~css
/*表示所有边框的的角的边长都是10*/
border-radius： 10px;
/*表示左上和右下是10px*/
border-radius： 10px 20px;
/**/
border-radius： 10px 20px 30px;
/**/
border-radius： 10px 20px 30px 40px;
~~~
注意：这里的2个值和上面的不一样，这个是对角线是一对的，而上面是左右是一对；
示例2：
~~~css
/* /表示是4个属性的前半和后半 */
border-radius: 4px 3px 6px / 2px 4px;
/*
表示：
border-top-left-radius: 4px 2px; 
border-top-right-radius: 3px 4px; 对角线
border-bottom-right-radius: 6px 2px; 
border-bottom-left-radius: 3px 4px; 对角线
*/
~~~

#### #background背景
8个属性：background-color,nackground-image,background-repeat,background-position,background-attchment,background-size,background-origin,background-clip
说明：值的顺序比较灵活，但又遵循一定的习惯，background-size一定要跟在background-position后面，用/分隔
示例：
~~~css
/*包含照片和颜色的复杂背景*/
background:
	red
	url('image.png')
	no-repeat
	fixed
	center / cover; 
/*注意中间没有分号，最后面才有*/
~~~

#### #text文本
6个属性：font-style,font-variant,font-weight,font-size,line-height,font-family
说明：严格的，font-size和font-family是必须的，line-height必须跟在font-height后面，用/分隔
语法：\[style] \[variant] \[weight] size\[/line-height] \[family]
示例
~~~css

/* 包含粗体和斜体 */ 
font: italic bold 1rem/1.6 "Helvetica Neue", Arial, sans-serif;
~~~
#### #flexbox布局
##### #flex
3个属性：flex-grow，flex-shrink，flex-basis
说明：这个是flexbox的核心缩写，用于子项目
示例
~~~css
.main {
  /* flex-grow: 2;
     flex-shrink: 1;
     flex-basis: auto; */
  flex: 2 1 auto;
}
~~~

##### #flex-flow
2个属性：flex-direction，flex-wrap
说明：用于flex容器，同时设置主轴方向和换行方向
~~~css
.main{
	/*主轴为行，允许换行，否则交换轴拉伸，主轴不拉伸*/
	flex-flow: row wrap;
}
~~~

#### #grid布局
##### #gap
2个属性：row-gap，column-gap
说明：可用于flexbox和grid
示例：
~~~css
gap: 20px; /* 行和列间距都是 20px */
gap: 10px 20px; /* 行间距 10px, 列间距 20px */
~~~

##### #place-items(用的少)
2个属性：align-content，justify-content
说明：在grid布局中用于同时设置垂直和水平对齐
示例：
~~~css
/* 简便的居中方法 */
place-items: center; /* 等同于 align-items: center; justify-items: center; */
~~~

##### #place-content(同上)
2个属性：align-content，justify-content
说明：用于对齐grid容器内的所有网格
示例：
~~~css
place-content: center;
~~~
