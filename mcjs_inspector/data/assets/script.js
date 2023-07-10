

addEventListener('DOMContentLoaded', (event) => {
    const valueElements = document.getElementsByClassName('value')

    function setHighlighted(valueId) {
        if (valueId === null) {
            for (const elm of valueElements)
                elm.classList.remove('highlighted')
        } else { 
            for (const elm of valueElements)
                if (elm.dataset.mcjsValue === valueId)
                    elm.classList.add('highlighted')
        }
    }
    
    for (const element of valueElements) {
        const valueId = element.dataset.mcjsValue;
        if (! /^[\w\d]+$/.test(valueId)) {
            console.warn(`${element}: invalid valueId: ${valueId}`)
            continue
        }

        element.onmouseenter = () => {
            setHighlighted(valueId)
        }
        element.onmouseleave = () => {
            setHighlighted(null)
        }
    }
})
