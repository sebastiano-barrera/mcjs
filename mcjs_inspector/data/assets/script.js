
class ScrollIntoViewInteraction {
    constructor(scrollArea) {
        this.scrollArea = scrollArea
        this.savedTop = null
        this.timeout = null
    }

    scrollTo(child) {
        if (this.savedTop === null) {
            this.savedTop = this.scrollArea.scrollTop
        }
        if (this.timeout !== null) {
            clearTimeout(this.timeout)
            this.timeout = null
        }

        child.scrollIntoView({
            block: 'center',
            behavior: 'smooth',
        })
    }

    resetLater() {
        if (this.timeout !== null)
            clearTimeout(this.timeout)

        this.timeout = setTimeout(() => {
            this.scrollArea.scrollTo({
                top: this.savedTop,
                behavior: 'smooth',
            })
            this.timeout = null
            this.savedTop = null
        }, 1000)
    }
}

document.body.addEventListener('htmx:load', (evt) => {
    const valueElements = document.getElementsByClassName('value')

    const stack = {
        scrollArea: document.getElementById('stack-scroll-area'),
        elementOfValue: new Map(),
    }
    for (const elm of stack.scrollArea.getElementsByClassName('value')) {
        const valueId = elm.dataset.mcjsValue
        stack.elementOfValue.set(valueId, elm)
    }

    const scrollIntoView = new ScrollIntoViewInteraction(stack.scrollArea)

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
                    scrollIntoView.scrollTo(stackElement)
            }
        }
        element.onmouseleave = () => {
            setHighlighted(null)
            scrollIntoView.resetLater()
        }
    }


})


