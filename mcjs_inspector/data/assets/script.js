
function isElementVisible(elm) {
    // Liberally inspired by https://stackoverflow.com/a/5354536
    const rect = elm.getBoundingClientRect();
    const viewHeight = Math.max(document.documentElement.clientHeight, window.innerHeight);
    return !(rect.bottom < 0 || rect.top >= viewHeight);
}

class ScrollIntoViewInteraction {
    DELAY_BEFORE_SCROLLING_BACK = 400

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
        if (this.savedTop === null)
            return;
        if (this.timeout !== null)
            clearTimeout(this.timeout)

        this.timeout = setTimeout(() => {
            this.scrollArea.scrollTo({
                top: this.savedTop,
                behavior: 'smooth',
            })
            this.timeout = null
            this.savedTop = null
        }, this.DELAY_BEFORE_SCROLLING_BACK)
    }
}

class VisibilityIndicator {
    constructor(element) {
        this.element = element
    }

    init() {
        this.element.addEventListener('click', () => {
            this.target.scrollIntoView({ behavior: 'smooth' })
        })
        this.update()
    }

    get target() {
        const targetId = this.element.dataset.visibilityTarget
        return document.getElementById(targetId)
    }

    update() {
        const target = this.target
        if (typeof target === 'object') {
            if (isElementVisible(target))
                this.element.classList.add('target-visible');
            else
                this.element.classList.remove('target-visible');
        }
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
    {
        // Scroll to the bottom at startup
        const elm = document.getElementById('code')
        elm.scrollTo({ top: elm.scrollHeight })
    }

    const indicators = []
    for (const elm of document.getElementsByClassName('script/visibility-indicator')) {
        const visind = new VisibilityIndicator(elm)
        visind.init()
        indicators.push(visind)
    }

    stack.scrollArea.onscroll = () => {
        for (const visind of indicators)
            visind.update()
    }

    const scrollIntoView = new ScrollIntoViewInteraction(stack.scrollArea)

    function setHighlighted(valueId) {
        const toast = document.getElementById('toast-past-call')
        toast.classList.add('hidden')

        if (typeof valueId === 'string') {
            let valuePresent = false
            for (const elm of valueElements) {
                if (elm.dataset.mcjsValue === valueId) {
                    elm.classList.add('highlighted')
                    if (stack.scrollArea.contains(elm))
                        valuePresent = true
                }
            }

            if (!valuePresent) {
                toast.classList.remove('hidden')
            }
        } else {
            for (const elm of valueElements) {
                elm.classList.remove('highlighted')
            }
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


