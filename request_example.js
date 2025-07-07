const images = []
fetch("https://img.freepik.com/free-photo/portrait-man-looking-front-him_23-2148422271.jpg?semt=ais_hybrid&w=740")
.then(res => res.blob())
.then(data => {
    const reader = new FileReader();
    reader.onload = () => {
        images.push(reader.result);
        fetch("http://localhost:5000/record", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                images: images,
                age: 25,
                score: 50
            })
        })
    };
    reader.readAsDataURL(data);
})
