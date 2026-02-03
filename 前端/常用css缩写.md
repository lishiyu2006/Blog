## 常用css属性缩写
### 对于和盒子四边相关的模型（margin和padding等）遵循时钟模型，即顺时针，上->右->下->左

- 4个值：上 右 下 左
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
注意：这里的2个值和上面的不一样，这个是对角线是一对的，第一个值控制“左上 & 右下”，第二个值控制“右上 & 左下”；

示例2：
~~~css
/* /表示是4个属性的前半和后半 */
border-radius: 4px 3px 6px / 2px 4px;
/*
表示：
border-top-left-radius: 4px 2px; 
border-top-right-radius: 3px 4px; 左上&右下
border-bottom-right-radius: 6px 2px; 
border-bottom-left-radius: 3px 4px; 右上&左下
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

#### #list-style 列表样式
3个属性：`list-style-type`, `list-style-position`, `list-style-image`
说明：常用于清除 `<ul>` 或 `<li>` 的默认圆点。
示例：
```CSS
/* 最常见的用法：清除默认样式 */
list-style: none;

/* 详细用法：方块类型 悬挂缩进 */
list-style: square inside;
```

#### #outline 轮廓
3个属性：`outline-width`, `outline-style`, `outline-color`
说明：与 `border` 类似，但不占据空间（不影响盒子模型布局），常用于 `input` 聚焦时的显示。
示例：
```CSS
outline: 2px solid blue;
outline: none; /* 清除聚焦时的蓝色外边框 */
```

#### #columns 多栏布局
2个属性：`column-width`, `column-count`
说明：用于将文本内容像报纸一样分成多列。
示例：
```CSS
/* 设置每栏最小宽度为 200px，并自动分成最大列数 */
columns: 200px;

/* 设置固定为 3 列 */
columns: 3;

/* 混合：最小宽度 250px，且最多 3 列 */
columns: 250px 3;
```

#### #inset 绝对定位缩写 (现代 CSS 推荐)
4个属性：`top`, `right`, `bottom`, `left`
说明：这是一个非常棒的新属性，它是 `top`, `right`, `bottom`, `left` 的缩写，逻辑遵循你提到的时钟模型。
示例：
```CSS
/* 将一个定位元素拉满整个父容器 */
position: absolute;
inset: 0; /* 等同于 top:0; right:0; bottom:0; left:0; */

/* 时钟模型应用 */
inset: 10px 20px 30px 40px; 
```

#### #flexbox布局
##### #flex
3个属性：flex-grow，flex-shrink，flex-basis
说明：这个是flexbox的核心缩写，用于**子项目**
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
说明：用于**flex容器**，同时设置主轴方向和换行方向
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

#### #transition过渡
4个属性：transition-property, transition-duration, transition-timing-function, transition-delay
说明：这是最常用的动画缩写，用于设置元素状态变化时的平滑过渡。
示例：
```CSS
/* 属性名 持续时间 动画曲线 延迟时间 */
transition: all 0.3s ease-in-out 0.1s;

/* 也可以只写前两个 */
transition: transform 0.2s;
```

##### #animation动画帧
8个属性：name，duration，timing-function，delay，iteration-count，direction，fill-mode，play-state
说明：可以省略部分属性
示例：
~~~css
animation: move 2s infinite alternate ease-in-out;
/*mave是名字，时间是2喵，动画次数是重复，方向是交替，运动路线是ease-in-out*/
~~~
