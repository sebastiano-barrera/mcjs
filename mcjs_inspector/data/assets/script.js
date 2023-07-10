

addEventListener('DOMContentLoaded', (event) => {
    const valueElements = document.getElementsByClassName('value')

    const stack = {
        scrollArea: document.getElementById('stack-scroll-area'),
        elementOfValue: new Map(),
        restoreScroll: {
            top: null,
            timeoutId: null,
        },

        scrollTo(child) {
            if (this.restoreScroll.timeoutId !== null)
                clearTimeout(this.restoreScroll.timeoutId);
            this.restoreScroll.top = this.scrollArea.scrollTop
            child.scrollIntoView({
                block: 'center',
                behavior: 'smooth'
            })
        },

        unlockScroll() {
            if (this.restoreScroll.timeoutId !== null) {
                clearTimeout(this.restoreScroll.timeoutId)
                this.restoreScroll.timeoutId = null
            }
            
            this.restoreScroll.timeoutId = setTimeout(() => {
                stack.scrollArea.scrollTo({
                    top: stack.restoreScrollTop,
                    behavior: 'smooth'
                })
                stack.restoreScrollTop = null
            }, 1000)
        },
    }
    for (const elm of stack.scrollArea.getElementsByClassName('value')) {
        const valueId = elm.dataset.mcjsValue
        stack.elementOfValue.set(valueId, elm)
    }

    function setHighlighted(valueId) {
        if (typeof valueId === 'string') {
            for (const elm of valueElements)
                if (elm.dataset.mcjsValue === valueId)
                    elm.classList.add('highlighted')
        } else {
            for (const elm of valueElements)
                elm.classList.remove('highlighted')
        }
    }

    for (const element of valueElements) {
        const valueId = element.dataset.mcjsValue;
        if (! /^[-\w\d]+$/.test(valueId)) {
            console.warn(`${element}: invalid valueId: ${valueId}`)
            continue
        }

        element.onmouseenter = (event) => {
            setHighlighted(valueId)
            if (!stack.scrollArea.contains(event.target)) {
                const stackElement = stack.elementOfValue.get(valueId)
                if (stackElement !== undefined)
                    stack.scrollTo(stackElement)
            }
        }
        element.onmouseleave = () => {
            setHighlighted(null)
            stack.unlockScroll()
        }
    }


})


