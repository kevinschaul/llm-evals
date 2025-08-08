// Function for use in transformResponse that throws an error when the
// http request is not successful
export default function (json, text, context) {
  if (context.response.status === 200) {
    return json || text
  } else {
    throw new Error("Server error")
  }
}
