# Twin-chat-V1
Backend
Step-1: /query_clusters

curl -X 'POST' \
  'http://127.0.0.1:8000/query_clusters' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Millennials with high auto debt and sub-prime credit.",
  "top_k": 3
}'
  {
  "query": "Millennials with high auto debt and sub-prime credit.",
  "top_k": 3
}  [
  {
    "cluster_id": 2,
    "cluster_name": "Affluent Asset Maximizers",
    "similarity": 0.751,
    "summary": "A young adult single household in the USA, with a low credit score in the subprime first quartile, holds a high equity balance and has the lowest numb..."
  },
  {
    "cluster_id": 6,
    "cluster_name": "Slowing Super-Prime",
    "similarity": 0.734,
    "summary": "A young adult, living in the USA, earns between $75,001 and $100,000, and is relatively new to credit with a low equity balance and a sub-prime credit..."
  },
  {
    "cluster_id": 3,
    "cluster_name": "Mid-Range Auto-Focused",
    "similarity": 0.732,
    "summary": "A 25-34-year-old Hispanic woman falls into the Established Consumer category, utilizing moderate credit usage. ..."
  }
]

Step-2: /cluster/{cluster_id}/personas
 curl -X 'GET' \
  'http://127.0.0.1:8000/cluster/6/personas' \
  -H 'accept: application/json'

cluster_id * 6
integer
  [
  {
    "persona_id": 94,
    "persona_name": "Charlotte",
    "gender": "Female",
    "age": "18 - 24",
    "summary": "A young adult, living in the USA, earns between $75,001 and $100,000, and is relatively new to credit with a low equity balance and a sub-prime credit score (1st quartile). This single household frequ"
  },
  {
    "persona_id": 95,
    "persona_name": "Mia",
    "gender": "Female",
    "age": "25 - 34",
    "summary": "A professional single family, primarily based in the USA, falls into the Established Consumers category. They have a moderate credit usage, a high equity balance, and a discretionary spend of $75,001 "
  },
  {
    "persona_id": 96,
    "persona_name": "Sophia",
    "gender": "Female",
    "age": "35 - 44",
    "summary": "A 50-year-old American with a high school education and net worth under $1 serves as primary decision-maker for an established household. Their net worth places them in the first decile, and they have"
  },
  {
    "persona_id": 97,
    "persona_name": "Amelia",
    "gender": "Female",
    "age": "45 - 54",
    "summary": "This USA-based individual, a primary decision-maker with a low credit score in the Sub Prime - 1st Quartile, has an income between $100,001 and $200,000. They have the fewest active open auto loans an"
  },
  {
    "persona_id": 98,
    "persona_name": "Isabella",
    "gender": "Female",
    "age": "55 - 64",
    "summary": "This group consists of high-earning, near-prime to prime singles or families, with a net worth between $400,001 and $600,001. They are moderate credit users, owning fewer than average active open auto"
  },
  {
    "persona_id": 99,
    "persona_name": "Isabella",
    "gender": "Female",
    "age": "65 - 74",
    "summary": "This single, elderly consumer, part of the Up & Coming Consumers group, has a low average credit score and falls into the Sub Prime - 1st Quartile category. With a net worth between $100,001 and $200,"
  },
  {
    "persona_id": 100,
    "persona_name": "Amelia",
    "gender": "Female",
    "age": "75+",
    "summary": "This group consists of elderly college graduates, mostly single family units, with a wealth rating in the sixth decile and an income between $400,001 and $600,001. They are active credit users with hi"
  },
  {
    "persona_id": 101,
    "persona_name": "Liam",
    "gender": "Male",
    "age": "18 - 24",
    "summary": "This young adult, a prime to super prime consumer with a high credit score and a six-figure income, frequently uses credit and has a low auto loan balance. Their spending habits include frequent purch"
  },
  {
    "persona_id": 102,
    "persona_name": "Benjamin",
    "gender": "Male",
    "age": "25 - 34",
    "summary": "This high-income, single family group consists of prime to super prime graduates with an average credit score in the fourth quartile. They are active credit users and have a high spending capacity, to"
  },
  {
    "persona_id": 103,
    "persona_name": "Mason",
    "gender": "Male",
    "age": "25 - 34",
    "summary": "A 25-35 year-old, single, other ethnicity professional resides in the USA. With some college education, this individual is in the second quartile of the Sub Prime to Near Prime credit tier and has a n"
  },
  {
    "persona_id": 104,
    "persona_name": "Henry",
    "gender": "Male",
    "age": "35 - 44",
    "summary": "This single, up-and-coming consumer is a high-earning individual, falling into the 6th decile wealth rating with an income between $100,001 and $200,000. They are relatively new to credit and have a h"
  },
  {
    "persona_id": 105,
    "persona_name": "Logan",
    "gender": "Male",
    "age": "45 - 54",
    "summary": "A single, college-educated American resides in the highest credit score quartile, holding a lien-free or open lien status. This individual is a management professional with a discretionary spend of $6"
  },
  {
    "persona_id": 106,
    "persona_name": "Henry",
    "gender": "Male",
    "age": "55 - 64",
    "summary": "A 55-64-year-old American male, with unknown religion and ethnicity, falls into the Equity Balance High and Established Consumer category. He moderately uses credit."
  },
  {
    "persona_id": 107,
    "persona_name": "Liam",
    "gender": "Male",
    "age": "65 - 74",
    "summary": "A high-spending, near prime to prime male consumer, aged 3rd quartile, earns between $200,001 and $400,000. He actively uses credit and donates over $50 to political causes annually. This group also c"
  },
  {
    "persona_id": 108,
    "persona_name": "James",
    "gender": "Male",
    "age": "75+",
    "summary": "An elderly consumer, residing in the USA, falls into the Established Consumers category. With a household income between $100,001 and $200,000, this individual has a high equity balance and is a moder"
  }
]
Response headers  Step-3: “/cluster/{cluster_id}/refine_personas”
{
  "query": "25 to 34 year old residing in USA",
  "top_k": 5
}
[
  {
    "persona_id": 103,
    "persona_name": "Mason",
    "gender": "Male",
    "age": "25 - 34",
    "summary": "A 25-35 year-old, single, other ethnicity professional resides in the USA. With some college education, this individual is in the second quartile of the Sub Prime to Near Prime credit tier and has a n"
  },
  {
    "persona_id": 95,
    "persona_name": "Mia",
    "gender": "Female",
    "age": "25 - 34",
    "summary": "A professional single family, primarily based in the USA, falls into the Established Consumers category. They have a moderate credit usage, a high equity balance, and a discretionary spend of $75,001 "
  },
  {
    "persona_id": 102,
    "persona_name": "Benjamin",
    "gender": "Male",
    "age": "25 - 34",
    "summary": "This high-income, single family group consists of prime to super prime graduates with an average credit score in the fourth quartile. They are active credit users and have a high spending capacity, to"
  }
]

Step-4: "/persona/{persona_id}/start_chat"

curl -X 'GET' \
  'http://127.0.0.1:8000/persona/103/start_chat' \
  -H 'accept: application/json'
persona_id *
integer.       103

(path)

{
  "persona_id": 103,
  "persona_name": "Mason",
  "cluster_name": "Slowing Super-Prime",
  "greeting": "👋 Hi, I’m Mason. I’m part of the **Slowing Super-Prime** cluster — a 25 - 34-year-old male who behaves like this: A 25-35 year-old, single, other ethnicity professional resides in the USA. With some college education, this individual is in the second quartile of the Sub Prime to Near Prime cre... Nice to meet you!"
}

Step-5:/persona/{persona_id}/chat
 persona_id *
integer               103

(path)
{
  "persona_id": 103,
  "reply": " I currently live in the United States of America."
}

 {
  "message": "Where do you reside Mason?"
}
